import warnings
import pandas as pd
import numpy as np

import cvxpy
import scipy

import time

from ..utils.verbose_print import verbose_init

class GeoDistributor:
    '''Class that handles the distribution of animals and crops across regions for a given demand (D) and a number of constraints
    by minimising deviation from an initial distribution of animal heads and crop areas (x0) .
    
    Parameters
    ----------
    x0 : dict of pandas.Series
        A dict of pandas series representing emand for animald (D['ani']) and crop products (D['crp'])
        x0['ani'].index should be on the form (species, breed, production system, region)
        x0['crp'].index should be on the form (*land use*, *crop group* ,crop, production system, region)
    demand : DemandAndConversions object
    herds : pandas.Series of AnimalHerd objects
    crops : CropProduction object
        
    Attributes
    ----------'''

    def __init__(self,x0,demand,crops,herds,feed_mgmt):
        
        
        self.x0 = {k:v.copy() for k,v in zip(x0.keys(),x0.values())}

        self.demand = demand
        self.crops = crops
        self.herds = herds
        self.feed_mgmt = feed_mgmt

        # Define index for x
        self.x_idx = {
            'ani' : pd.MultiIndex.from_tuples(
                    [(sp,br,ps,ss,re) for (sp,br,ps,ss) in self.herds.index for re in self.herds[(sp,br,ps,ss)].index],
                    names=['species','breed','prod_system','sub_system','region']
                    ),
            'crp' : self.crops.index
        }

        # Sort x0['ani'] to match x['ani']
        self.x0['ani'] = self.x0['ani'].loc[self.x_idx['ani'].droplevel('sub_system').unique()]
       
        # Store x0 indexes
        self.x0_idx = {
            'ani' : self.x0['ani'].index,
            'crp' : self.x0['crp'].index
        }

    def make(self, use_cons='all', scale_power=[0,0], verbose=False):
        '''Creates constraints and defines optimisation problem'''

        vprint = verbose_init(verbose, id_str='GeoDist.make')

        vprint('Creating demand vector ...')
        self.get_demand()

        vprint('Scaling ...')
        # Calculate scaling factors
        self.calculate_scaling_factors(scale_power)

        # Apply scaling factors to x0
        self.x0s = {
            key : df * self.scale_f[key]
            for key,df
            in zip(self.x0.keys(),self.x0.values())
            }

        if use_cons == 'all':
            use_cons = [1,2,3,4,5,6,7]
        elif not isinstance(use_cons,list):
            use_cons = [use_cons]

        use_cons = [str(e) for e in use_cons]

        # Make constraints
        for nr in use_cons:
            fun = getattr(self,'make_C'+nr)
            vprint(f'Making constraint C{nr} ...')
            fun()

        # Make objective function(s)
        vprint('Making objective O1 ...')
        self.make_O1()

        vprint('Defining problem ...')
        self.define_cvx_problem(use_cons)

        vprint(type='end')

    def solve(self, solver_settings='default', verbose=False):

        vprint = verbose_init(verbose, id_str='GeoDist.solve')

        # Default solver settings
        #
        # OSQP
        # ----
        # Settings for OSQP available at https://osqp.org/docs/interfaces/solver_settings.html
        # Using a too high tolerance (eps_abs, eps_rel) leads to large relative deviations
        # from x0 for crops with small areas, but a low tolerance increases time to find solution.
        if solver_settings=='default':
            solver_settings = {
                'solver':'OSQP',
                'max_iter':200000,
                'eps_abs':1e-6,
                'eps_rel':1e-6,
                'verbose':False
            }
        
        # If a list of alternative solver/settings is not supplied
        # make a one element list
        if not isinstance(solver_settings, list):
            solver_settings = [solver_settings]
        
        vprint('Finding solution ...')
        
        # Try to find a solution with (potentially) different solver/settings
        # If an optimal solution is found break and do not try next solver/settings
        for kwargs in solver_settings:
            solver = kwargs['solver']
            self.problem.solve(**kwargs)

            if 'optimal' in self.problem.status:
                break

        # Check solution and print results    
        if 'optimal' in self.problem.status:

            # DO SOME MORE FEASIBILITY CHECKS ON THE SOLUTION HERE!!

            vprint(f'Optimal solution found! Status: \'{self.problem.status}\', Solver: \'{solver}\'')
            self.success = False

            # Get and store optimal value for variable
            xs = self.problem.variables()[0].value
            self.xs = {
                'ani' : pd.Series(
                    xs[:len(self.x_idx['ani'])],
                    index = self.x_idx['ani']
                ),
                'crp' : pd.Series(
                    xs[len(self.x_idx['ani']):],
                    index = self.x_idx['crp']
                )
            }

            self.x = {
                key : (
                    (df / self.scale_f[key])
                    .reorder_levels(self.x_idx[key].names)
                    .reindex(self.x_idx[key])
                )
                for key,df
                in zip(self.xs.keys(),self.xs.values())
                }
        else:
            vprint(f'No solution found!')
            self.success = False
            # NEED TO IMPLEMENT A WAY TO HANDLE THIS SITUATION

        vprint(type='end')

    def get_demand(self):
        self.D = {
            'ani' : self.demand.animal_prod_demand.copy(),
            'crp' : self.demand.crop_prod_demand.sum(axis=1)
            }
        
        # Add rows for any domestically produced crop products used for feed or seed not already in crop product demand vector (D['crp'])
        self.feed_mgmt.par.clear()
        for cp in set(self.feed_mgmt.par.get_unique('crop_prod')) | set(self.crops.par.get_unique('crop_prod', qry='parameter == "seed"')):
            for ps in self.D['crp'].index.get_level_values('prod_system').unique():
                idx = (ps,cp)
                if (self.feed_mgmt.par.get('share_imported', crop_prod=cp, prod_system=ps) != 100).any() & (idx not in self.D['crp'].index):
                    self.D['crp'][idx] = 0

        # Store indexes
        self.D_idx = {
            'ani' : self.D['ani'].index,
            'crp' : self.D['crp'].index
        }

    def calculate_scaling_factors(self,scale_power=[0,0]):

        scale_f = {key:df.copy() for key,df in zip(self.x0.keys(),self.x0.values())}

        # Get x0 for animals and combine index levels crop+production system
        x0_ani = self.x0['ani'].to_frame()
        x0_ani['item'] = ['_'.join([sp,br,ps]) for sp,br,ps in self.x0['ani'].index.droplevel('region')]
        x0_ani = x0_ani.set_index('item', append=True).droplevel(['species','breed','prod_system'])[self.x0['ani'].name]

        # Get x0 for crops and combine index levels species+breed+production system
        x0_crp = self.x0['crp'].to_frame()
        x0_crp['item'] = ['_'.join([cr,ps]) for cr,ps in self.x0['crp'].index.droplevel('region')]
        x0_crp = x0_crp.set_index('item', append=True).droplevel(['crop','prod_system'])[self.x0['crp'].name]

        x0_ = pd.concat((x0_ani,x0_crp))

        # Calculate scale factor with regards to item (i.e. species+breed+production system or crop+production system)
        sums1 = x0_.groupby('item').transform('mean')
        f1 = (sums1.mean() / sums1) * scale_power[0] + 1 * (1-scale_power[0])
        f1[f1==np.inf] = f1[f1!=np.inf].mean() # temp fix
        
        # Calculate scale factor with regards to region 
        sums2 = x0_.groupby('region').transform('mean')
        f2 = (sums2.mean() / sums2) * scale_power[1] + 1 * (1-scale_power[1])

        # Assert results
        assert np.isfinite(f1).all()
        assert np.isfinite(f2).all()

        # Calculate final scale factors
        f = (f1 * f2)

        scale_f['ani'].iloc[:] = f[:len(scale_f['ani'])]
        scale_f['crp'].iloc[:] = f[len(scale_f['ani']):]
        self.scale_f = scale_f


    def define_cvx_problem(self,use_cons):
        
        # Get scaling factors
        sf = np.concatenate([self.scale_f[k].reindex(self.x_idx[k], method='ffill') for k in ['ani','crp']])

        xs = cvxpy.Variable(len(self.x_idx['ani'])+len(self.x_idx['crp']), nonneg=True)

        O1 = cvxpy.sum_squares(self.P1.M @ xs - np.concatenate((self.x0s['ani'].values,self.x0s['crp'].values)))
        OBJ = cvxpy.Minimize(O1)

        CONS = []
        if '1' in use_cons:
            CONS.append(self.A1.M @ (xs/sf) == self.b1)
        if '2' in use_cons:
            CONS.append(self.A2.M @ (xs/sf) >= 0)
        if '3' in use_cons:
            CONS.append(self.A3.M @ (xs/sf) <= self.b3)
        if '4' in use_cons:
            CONS.append(self.A4.M @ (xs/sf) <= self.b4)
        if '5' in use_cons:
            pass
        if '6' in use_cons:
            CONS.append(self.A6.M @ (xs/sf) <= 0)
        if '7' in use_cons:
            pass

        # Define problem
        self.problem = cvxpy.Problem(
            objective = OBJ,
            constraints = CONS
        )

    def make_C1(self):
        '''Creates A-matrix for constraint (C1):
        Production must meet demand A1 @ x == b1, where b1 is national demand per animal/crop product (D)'''

        # Animal product demand
        A1_1 = self.make_A1_1()
        # Feed demand
        A1_2 = self.make_A1_2()
        # Crop product demand
        A1_3 = self.make_A1_3()

        # Stack matrices
        A1 = scipy.sparse.vstack([
            scipy.sparse.hstack([
                A1_1.M,
                scipy.sparse.csc_matrix((A1_1.M.shape[0],A1_3.M.shape[1]))
            ]),
            scipy.sparse.hstack([
                A1_2.M,
                A1_3.M
            ])
        ])

        self.A1 = IndexedMatrix(
            matrix=A1,
            row_idx={'ani':A1_1.rows, 'crp':A1_2.rows},
            col_idx={'ani':A1_1.cols, 'crp':A1_3.cols}
        )
        self.b1 = np.concatenate((self.D['ani'].values,self.D['crp'].values))

    def make_C2(self):
        '''Creates C2'''

        # Regional feed demand for crop products
        A2_1 = self.make_A2_1()
        # Production of crop products
        A2_2 = self.make_A2_2()

        # Stack matrices
        A2 = scipy.sparse.hstack([A2_1.M,A2_2.M])

        self.A2 = IndexedMatrix(
            matrix=A2,
            row_idx=A2_1.rows,
            col_idx={'ani':A2_1.cols, 'crp':A2_2.cols}
        )

    def make_C3(self):
        '''Creates C3: A3 @ x <= b3'''

        self.A3 = self.make_A3_and_A4(land_use='cropland')
        
        self.b3 = self.A3.M @ np.concatenate((np.zeros(len(self.x_idx['ani'])),self.x0['crp'])) * 1.1 # NOTE: Implement cropland area limit factor from parameter, how to deal with scaling here

    def make_C4(self):
        '''Creates C4: A4 @ x <= b4'''

        self.A4 = self.make_A3_and_A4(land_use='semi-natural grasslands')
        self.b4 = self.A4.M @ np.concatenate((np.zeros(len(self.x_idx['ani'])),self.x0['crp'])) * 1.1 # NOTE: Implement SNG area limit factor from parameter, how to deal with scaling here

    def make_C6(self):
        '''Creates C6: A6 @ x <= 0'''

        self.A6 = self.make_A6()


    def make_O1(self):
        
        # x['ani'] --> x0['ani']
        P1_1 = self.make_P1_1()
        # x['crp'] --> x0['crp']
        P1_2 = self.make_P1_2()

        P1 = scipy.sparse.vstack([
            scipy.sparse.hstack([
                P1_1.M,
                scipy.sparse.csc_matrix((P1_1.M.shape[0],P1_2.M.shape[1]))
            ]),
            scipy.sparse.hstack([
                scipy.sparse.csc_matrix((P1_2.M.shape[0],P1_1.M.shape[1])),
                P1_2.M
            ])
        ])

        self.P1 = IndexedMatrix(
            matrix=P1,
            row_idx={'ani':P1_1.rows, 'crp':P1_2.rows},
            col_idx={'ani':P1_1.cols, 'crp':P1_2.cols}
        )
        

    def make_A1_1(self):

        # Get row index from animal product demand vector (ps,sp,ap)
        row_idx = self.D_idx['ani']
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx['ani']

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        # Go through animal herds
        for herd in self.herds:

            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Go through animal products
            for ap in herd.production.columns.get_level_values('animal_prod').unique():
                # Go through output production systems
                for ops in herd.production.columns.get_level_values('prod_system').unique():
                    if (ops,sp,ap) in row_idx:
                        # Get production of animal product (ap) from output production system (ops) per head
                        # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                        # and region (re)
                        res = herd.production.loc[:,(ops,slice(None),ap)].sum(axis=1)

                        # Store values and row/col nr
                        val.extend(res.values)
                        row_nr.extend(
                            [row_idx.get_loc((ops,sp,ap))] * len(res)
                        )
                        col_nr.extend(
                            [col_idx.get_loc((sp,br,ps,ss,re)) for re in res.index]
                        )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_A1_2(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx['crp']
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx['ani']

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        # Go through animal herds
        for herd in self.herds:

            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Get crop products and production systems with domestic demand from feed
            opss_cps = (
                herd.feed.crop_product_demand
                .xs('domestic', level='origin', axis=1)
                # drop feeds with no (< 5e-6 kg) domestic demand
                .round(5).replace({0:np.nan}).dropna(axis=1, how='all') 
                .droplevel('animal', axis=1)
                .columns
                .unique()
                .values
            )

            # Go through crop products and production systems used as feed
            for ops,cp in opss_cps:
                # Get feed demand for crop product (cp) from output production system (ops) per head
                # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                # and region (re)
                res = - herd.feed.crop_product_demand.loc[:,('domestic',ops,slice(None),cp)].sum(axis=1)

                # Store values and row/col nr
                val.extend(res.values)
                row_nr.extend(
                    [row_idx.get_loc((ops,cp))] * len(res)
                )
                col_nr.extend(
                    [col_idx.get_loc((sp,br,ps,ss,re)) for re in res.index]
                )
        
        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_A1_3(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx['crp']
        # Get col index from crop production (cr,ps,re)
        col_idx = self.x_idx['crp']

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        cps = self.crops.production.columns

        for cr,ps in col_idx.droplevel('region').unique():
            for cp in cps:
                if (ps,cp) in row_idx:
                    # Get production of crop product (cp) minus seed demand
                    # from production system (ps) per area of crop (cr)
                    # in production system (ps) and region (re)
                    res = (
                        self.crops.production.loc[(cr,ps,slice(None)),(cp)]
                        - (self.crops.seed_demand.loc[(cr,ps,slice(None)),(cp)] if cp in self.crops.seed_demand.columns else 0)
                    )

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [row_idx.get_loc((ps,cp))] * len(res)
                    )
                    col_nr.extend(
                        [col_idx.get_loc((cr,ps,re)) for re in res.index.get_level_values('region')]
                    )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_A2_1(self):
        # Get row index from feeds with a regional demand (cp,ps,re)
        row_idx = pd.MultiIndex.from_tuples([
            (cp,ps,re)
            for cp in list(set([
                                cp
                                for herd in self.herds if 'regional' in herd.feed.crop_product_demand.columns
                                for cp in (
                                    herd.feed.crop_product_demand['regional']
                                    .replace({0:np.nan}).dropna(axis=1, how='all') # drop feeds with no regional demand
                                    .columns.get_level_values('crop_prod')
                                )
                            ]))
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ], names = ['crop_prod','prod_system','region'])
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx['ani']

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        # Go through animal herds
        for herd in self.herds:

            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Check if herd has any regional demand for feeds
            if 'regional' in herd.feed.crop_product_demand.columns:

                # Get crop products and production systems with regional demand from feed
                opss_cps = (
                    herd.feed.crop_product_demand
                    .xs('regional', level='origin', axis=1)
                    .replace({0:np.nan}).dropna(axis=1, how='all') # drop feeds with no domestic demand
                    .droplevel('animal', axis=1)
                    .columns
                    .unique()
                    .values
                )

                # Go through crop products adn production systems with regional demand from feed
                for ops,cp in opss_cps:
                    # Get regional feed demand for crop product (cp) from output production system (ops) per head
                    # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                    # and region (re)
                    res = - herd.feed.crop_product_demand.loc[:,('regional',ops,slice(None),cp)].sum(axis=1)

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [row_idx.get_loc((cp,ops,re)) for re in res.index]
                    )
                    col_nr.extend(
                        [col_idx.get_loc((sp,br,ps,ss,re)) for re in res.index]
                    )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_A2_2(self):
        # Get row index from feeds with a regional demand (cp,ps,re)
        row_idx = pd.MultiIndex.from_tuples([
            (cp,ps,re)
            for cp in list(set([
                                cp
                                for herd in self.herds if 'regional' in herd.feed.crop_product_demand.columns
                                for cp in (
                                    herd.feed.crop_product_demand['regional']
                                    .replace({0:np.nan}).dropna(axis=1, how='all') # drop feeds with no regional demand
                                    .columns.get_level_values('crop_prod')
                                )
                            ]))
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ], names = ['crop_prod','prod_system','region'])
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        row_idx_lookup = row_idx.droplevel('region').sort_values()

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []


        for cr,ps in self.crops.production.index.droplevel('region')[
            self.crops.production[row_idx.get_level_values('crop_prod').unique()].sum(axis=1)>0
            ].unique():
            for cp in row_idx.get_level_values('crop_prod').unique():
                if (cp,ps) in row_idx_lookup:
                    # Get production of crop product (cp) from production system (ps) per area of crop (cr)
                    # in production system (ps) and region (re)
                    res = self.crops.production.loc[(cr,ps,slice(None)),(cp)].fillna(0)

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [row_idx.get_loc((cp,ps,re)) for re in res.index.get_level_values('region')]
                    )
                    col_nr.extend(
                        [col_idx.get_loc((cr,ps,re)) for re in res.index.get_level_values('region')]
                    )
                    
        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_A3_and_A4(self,land_use):

        # Get row index from regions (re)
        row_idx = self.x_idx['crp'].get_level_values('region').unique()
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        # Get dict for translating crop --> land use
        rel = self.crops.par.get_rel('crop','land_use')

        # Data and corresponding row/col numbers for constructing matrix
        val = [1 if rel[cr] == land_use else 0 for cr,_,_ in col_idx]
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((re)) for _,_,re in col_idx]

        M = scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc()
        Z = scipy.sparse.csc_matrix((M.shape[0],len(self.x_idx['ani']))) # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z,M]),
            row_idx,
            {'ani':self.x_idx['ani'],'crp':col_idx}
        )

        return M
    
    def make_A6(self):
        '''Creates A-matrix for C6:
        
        Constrain the maximum share of cropland devoted to a given crop group
        in a given region in a given production system.

        Note to future:
        - Would it be usefull with a constraint for minimum share?
        - Deal with crops assumed not to be in rotation by putting 0 in the matrix'''

        self.crops.par.clear()

        # Get crop groups with max/min inclusion in rotation constraint
        cgs = self.crops.par.get_unique('crop_group', qry=f'parameter == "max_in_rot"')
        pss = self.x_idx['crp'].get_level_values('prod_system').unique()
        res = self.x_idx['crp'].get_level_values('region').unique()

        # Get row index from (cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [(cg,ps,re) for cg in cgs for ps in pss for re in res],
            names = ['crop_group','prod_system','region']
        )
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        # Get dict for translating crop --> land use
        lu_rel = self.crops.par.get_rel('crop','land_use')
        # Get dict for translating crop --> crop group
        cg_rel = self.crops.par.get_rel('crop','crop_group')

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for cg,ps in row_idx.droplevel('region').unique():
            f = float(self.crops.par.get('max_in_rot' ,crop_group=cg, prod_system=ps)/100)

            vls = [0 if (ps != ps_) | (lu_rel[cr] != 'cropland') else ((1-f) if cg_rel[cr] == cg else -f) for cr,ps_,_ in col_idx]
            cns = list(range(len(col_idx)))
            rns = [row_idx.get_loc((cg,ps,re)) for _,_,re in col_idx]

            val.extend(vls)
            col_nr.extend(cns)
            row_nr.extend(rns)

        M = scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc()
        Z = scipy.sparse.csc_matrix((M.shape[0],len(self.x_idx['ani']))) # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z,M]),
            row_idx,
            {'ani':self.x_idx['ani'],'crp':col_idx}
        )

        return M

    def make_P1_1(self):
        # Get row index from x0['ani'] (sp,br,ps,re)
        row_idx = self.x0_idx['ani']
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx['ani']

        # Data and corresponding row/col numbers for constructing matrix
        val = [1]*len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((sp,br,ps,re)) for sp,br,ps,_,re in col_idx]
        
        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

    def make_P1_2(self):
        '''Creates P1,2 matrix. That is an identity matrix of size len(x0['crp']) as x0['crp'].index == x['crp'].index '''
        # Get row index from x0['crp'] (cr,ps,re)
        row_idx = self.x0_idx['crp']
        # Get row index from x['crp'] (cr,ps,re)
        col_idx = self.x_idx['crp']

        # To store data and corresponding row/col numbers for constructing matrix
        val = [1]*len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((cr,ps,re)) for cr,ps,re in col_idx]
        
        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc(),
            row_idx,
            col_idx
        )

        return M

class IndexedMatrix():
    '''Class to store pandas.Index/MultiIndex alongside a sparse matrix to keep track of things'''

    def __init__(self,matrix,row_idx,col_idx):
        self.M = matrix

        if isinstance(row_idx,list):
            levels = list(row_idx[0].names)
            for idx in row_idx:
                add = [l for l in idx.names if l not in levels]
                levels.extend(add)
            print(levels)
            
        self.rows = row_idx
        self.cols = col_idx

    def eval(self, x):
        return pd.Series(self.M @ x, index=self.rows)