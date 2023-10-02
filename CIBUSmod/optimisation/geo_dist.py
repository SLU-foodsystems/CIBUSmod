import warnings
import pandas as pd
import numpy as np

import cvxpy
import scipy

import time

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned
from ..utils.output_data_manip import concat_herds

class GeoDistributor:
    '''Class that handles the distribution of animals and crops across regions for a given demand (D) and a number of constraints
    by minimising deviation from an initial distribution of animal heads and crop areas (x0) .
    
    Parameters
    ----------
    par : ParameterRetriever object
    x0 : dict of pandas.Series
        A dict of pandas series representing emand for animald (D['ani']) and crop products (D['crp'])
        x0['ani'].index should be on the form (species, breed, production system, region)
        x0['crp'].index should be on the form (*land use*, *crop group* ,crop, production system, region)
    demand : DemandAndConversions object
    herds : pandas.Series of AnimalHerd objects
    crops : CropProduction object
        
    Attributes
    ----------'''

    def __init__(self,par,regions,demand,crops,herds,feed_mgmt):
        
        self.par = par

        self.regions = regions
        self.demand = demand
        self.crops = crops
        self.herds = herds
        self.feed_mgmt = feed_mgmt

    def make(self, use_cons='all', scale_power=[0,0], verbose=False, **kwargs):
        '''Creates constraints and defines optimisation problem'''

        vprint = verbose_init(verbose, id_str='GeoDistributor.make')

        if use_cons == 'all':
            use_cons = [1,2,3,4,5,6,7]
        elif not isinstance(use_cons,list):
            use_cons = [use_cons]
        use_cons = [str(e) for e in use_cons]
        # Make sure that C7 is handled last
        use_cons.append(use_cons.pop(use_cons.index('7')))

        vprint('Getting x0 and making indexes ...')
        self.get_x0()

        vprint('Creating demand vector ...')
        self.get_demand()

        vprint('Scaling ...')
        # Calculate scaling factors
        self.calculate_scaling_factors(scale_power)
       
        # Make objective function(s)
        vprint('Making objective O1 ...')
        self.make_O1()

        # Make constraints
        self.cons_add_exec = [] # List to store code sniplets for including constraints
        for nr in use_cons:
            fun = getattr(self,'make_C'+nr)
            vprint(f'Making constraint C{nr} ...')
            fun(
                **{k:v for k,v in kwargs.items() if 'C'+nr in k}
            )

        # If C7 not included no variables are dropped
        if '7' not in use_cons:
            self.x_idx_short = {'ani':self.x_idx['ani'].copy(), 'crp':self.x_idx['crp'].copy()}

        vprint('Defining problem ...')
        self.define_cvx_problem(use_cons)

        vprint(type='end')

    def solve(self, solver_settings='default', apply_solution=True, verbose=False):
        '''Solve optimisation problem'''

        vprint = verbose_init(verbose, id_str='GeoDistributor.solve')

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

            vprint(f'Optimal solution found! Status: \'{self.problem.status}\', Itterations: {self.problem.solver_stats.num_iters}, Solver: \'{self.problem.solver_stats.solver_name}\'')
            self.success = True

            # Get and store optimal value for variable
            x = self.problem.variables()[0].value
            # Put xs on short index (!= index if C7 is used) and reindex 
            self.x = {
                'ani' : pd.Series(
                    x[:len(self.x_idx_short['ani'])],
                    index = self.x_idx_short['ani']
                ).reindex(self.x_idx['ani'], fill_value=0),
                'crp' : pd.Series(
                    x[len(self.x_idx_short['ani']):],
                    index = self.x_idx_short['crp']
                ).reindex(self.x_idx['crp'], fill_value=0)
            }

            if apply_solution:
                vprint(f'Applying solution')
                self.apply_solution()

        else:
            vprint(f'No solution found!')
            self.success = False
            # NEED TO IMPLEMENT A WAY TO HANDLE THIS SITUATION

        vprint(type='end')

    def apply_solution(self, x=None):
        '''Update CropProduction and AnumalHerds according to found solution'''

        if x is None:
            x = self.x

        # Update CropProduction
        self.crops.scale(x['crp'])
        # Update AnumalHerds
        for h in self.herds:
            h.scale(
                x['ani'].loc[(h.species,h.breed,h.prod_system,h.sub_system)],
                x_is = h.x_is
            )
        
        # Allocate crop areas to uses
        self.allocate_crop_production_per_use()

    def get_x0(self):
        # Get x0
        self.x0 = {
            'ani' : self.regions.x0_animals.copy(),
            'crp' : self.regions.x0_crops.copy()
        }

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

    def get_demand(self):
        self.D = {
            'ani' : self.demand.animal_prod_demand.sum(axis=1),
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

    def calculate_scaling_factors(self,scale_power=0):
        '''Calculates scaling factor to apply to x and x0 in objective O1
        as f = (mean(x0)/x0) ^ scale_power. In cases where f = inf
        (i.e. x0 = 0) f is set to the maximum value  where f != inf.
        scale_power = 0 gives no scaling.
        '''

        scale_f = {key:df.copy() for key,df in zip(self.x0.keys(),self.x0.values())}

        x0 = pd.concat((self.x0['ani'],self.x0['crp']))

        f = x0.mean()/x0
        f.loc[f==np.inf] = f.loc[f!=np.inf].max()
        f = f ** scale_power
        assert np.isfinite(f).all()

        scale_f['ani'].iloc[:] = f[:len(scale_f['ani'])]
        scale_f['crp'].iloc[:] = f[len(scale_f['ani']):]
        self.scale_f = scale_f

    def define_cvx_problem(self,use_cons):
        
        # Apply scaling factors to x0
        x0s = cvxpy.Constant(
            np.concatenate([
                (self.x0[k] * self.scale_f[k])
                .reindex(self.x0_idx[k])
                for k in ['ani','crp']
            ])
        )
        
        # Get scaling factors for x
        sf = cvxpy.Constant(
            np.concatenate([
                (
                    self.scale_f['ani']
                    .reindex(self.x_idx['ani'].reorder_levels(['species','breed','prod_system','region','sub_system']))
                    .reindex(self.x_idx_short['ani'].reorder_levels(['species','breed','prod_system','region','sub_system']))
                ),
                self.scale_f['crp'].reindex(self.x_idx_short['crp'])
            ])
        )

        n = len(self.x_idx_short['ani'])+len(self.x_idx_short['crp'])
        x = cvxpy.Variable(n, nonneg=True)

        O1 = cvxpy.sum_squares(
            (self.P1.M @ cvxpy.multiply(sf,x)) - x0s
        )
        OBJ = cvxpy.Minimize(O1)

        # Append constraints
        CONS = []
        for ex in self.cons_add_exec:
            exec(ex)

        # Define problem
        self.problem = cvxpy.Problem(
            objective = OBJ,
            constraints = CONS
        )

    def make_C1(self):
        '''Creates C1: A1 @ x == b1
        Production must meet demand. A1 @ x gives production and b1 is
        national demand per animal/crop product (D)
        '''

        # Animal product demand
        self.A1_1 = self.make_A1_1()
        # Feed demand
        self.A1_2 = self.make_A1_2()
        # Crop product demand
        self.A1_3 = self.make_A1_3()

        # Stack matrices
        A1 = scipy.sparse.vstack([
            scipy.sparse.hstack([
                self.A1_1.M,
                scipy.sparse.csc_matrix((self.A1_1.M.shape[0],self.A1_3.M.shape[1]))
            ]),
            scipy.sparse.hstack([
                self.A1_2.M,
                self.A1_3.M
            ])
        ])

        self.A1 = IndexedMatrix(
            matrix=A1,
            row_idx={'ani':self.A1_1.rows, 'crp':self.A1_2.rows},
            col_idx={'ani':self.A1_1.cols, 'crp':self.A1_3.cols}
        )
        self.b1 = np.concatenate((self.D['ani'].values,self.D['crp'].values))

        # Append code to include constraint when defining cvx problem
        self.cons_add_exec.extend(['CONS.append(self.A1.M @ x == self.b1)'])

    def make_C2(self):
        '''Creates C2'''

        # Regional feed demand for crop products
        self.A2_1 = self.make_A2_1()
        # Production of crop products
        self.A2_2 = self.make_A2_2()

        # Stack matrices
        A2 = scipy.sparse.hstack([self.A2_1.M,self.A2_2.M])

        self.A2 = IndexedMatrix(
            matrix=A2,
            row_idx=self.A2_1.rows,
            col_idx={'ani':self.A2_1.cols, 'crp':self.A2_2.cols}
        )

        # Append code to include constraint when defining cvx problem
        self.cons_add_exec.extend(['CONS.append(self.A2.M @ x >= 0)'])

    def make_C3(self):
        '''Creates C3: A3 @ x <= b3'''

        self.A3 = self.make_A3()
        
        self.b3 = (
            self.A3.M @ 
            np.concatenate((
                np.zeros(len(self.x_idx['ani'])),
                self.x0['crp']
            ))
         ) * self.regions.par.get(
            'max_land_use_factor',
            **self.A3.rows.to_frame().to_dict('list')
        )

        # Append code to include constraint when defining cvx problem
        self.cons_add_exec.extend(['CONS.append(self.A3.M @ x <= self.b3)'])

    def make_C4(self):
        '''Not used'''
        pass

    def make_C5(self):
        '''Creates C5: A5 @ x <= 0'''
        # Maximum supply of crop product(s) from crop(s)
        self.A5_1 = self.make_A5_1()
        # Production of crop products
        self.A5_2 = self.make_A5_2()

        # Stack matrices
        A5 = scipy.sparse.hstack([self.A5_1.M,self.A5_2.M])

        self.A5 = IndexedMatrix(
            matrix=A5,
            row_idx=self.A5_1.rows,
            col_idx={'ani':self.A5_1.cols, 'crp':self.A5_2.cols}
        )

        # Append code to include constraint when defining cvx problem
        self.cons_add_exec.extend(['CONS.append(self.A5.M @ x <= 0)'])
    
    def make_C6(self):
        '''Creates C6: A6 @ x <= 0'''

        self.A6 = self.make_A6()

        # Append code to include constraint when defining cvx problem
        self.cons_add_exec.extend(['CONS.append(self.A6.M @ x <= 0)'])

    def make_C7(self):
        '''Creates C7: Constrain crops to certain regions based on minimum growing degree days (GDD5).
        Also constrains animals with regional demand for crops that can't be grown in a region'''

        # This constraint is not implemented as a constraint in the solver but instead dropps
        # variables representing crops or animals that can't be present in a region. 
        # IMPORTANT: This must be run after all other contraints have been defined!
        
        # Index of crops
        cr_idx = self.x_idx['crp']
        
        # Get allowed crop-region combinations (i.e. region GDD5 >= min_GDD5 for crop)
        self.crops.par.clear()
        self.regions.par.clear()
        sel_cr = cr_idx[
            self.regions.par.get('GDD5',**cr_idx.to_frame().to_dict('list'))
            >=
            self.crops.par.get('min_GDD5',**cr_idx.to_frame().to_dict('list'))
        ]
        
        # Index of animal herds
        an_idx = self.x_idx['ani']
        
        # Get crop products that CAN be produced in region
        sel_cp = (
            self.crops.production.loc[sel_cr]
            .stack()
            .groupby(['crop_prod','prod_system','region'])
            .sum()
            .replace({0:np.nan}).dropna()
            .index
        )
        
        # Index of crop products
        cp_idx = (
            self.crops.production
            .stack().droplevel('crop')
            .reorder_levels(['crop_prod','prod_system','region'])
            .index.unique()
        )
        # Get crop products that CAN'T be produced in region
        nsel_cp = cp_idx.difference(sel_cp)
        
        # List to populate with herds that can be in region
        # (i.e. with no regional demand for feeds that can't be 
        # produced in the region) 
        sel_an = []

        for h in self.herds:
            # Get crop products with a regional feed demand
            nsel_cp2 = (
                (h.feed.crop_product_demand
                .xs('regional',axis=1)
                .stack(['prod_system','crop_prod'])
                .sum(axis=1)
                .reorder_levels(['crop_prod','prod_system','region'])
                > 0)
                .replace({False:np.nan}).dropna()
                .index
            )
            
            # Get regions where herd has a regional demand
            # for a feed that can't be grown. 
            nsel_re = nsel_cp.intersection(nsel_cp2).get_level_values('region').unique()
            
            # Get regions where herd CAN be present
            sel_re = h.index.difference(nsel_re)

            # Add herds allowed to animal selection
            sp = h.species
            br = h.breed
            ps = h.prod_system
            ss = h.sub_system
            sel_an += [(sp,br,ps,ss,re) for re in sel_re]
        # To pandas MultiIndex
        sel_an = pd.MultiIndex.from_tuples(sel_an,names=['species','breed','prod_system','sub_system','region'])
        
        # Get variable positions not to drop
        isel_an = [an_idx.get_loc(s) for s in sel_an]
        isel_cr = [cr_idx.get_loc(s) + len(an_idx) for s in sel_cr]
        isel = isel_an + isel_cr
        
        # Store short index (i.e. index of variables after dropping)
        self.x_idx_short = {'ani':sel_an, 'crp':sel_cr}
        
        # Drop variables from objective and constraint matrices
        for mat in ['P1','A1','A2','A3','A4','A5','A6','A8']:
            try:
                A = getattr(self,mat)
            except AttributeError:
                continue
            else:
                A.M = A.M[:,isel]
                A.cols['ani'] = sel_an.copy()
                A.cols['crp'] = sel_cr.copy()

    def make_C8(
            self,
            C8_crp = None,
            C8_ani = None,
            C8_rel = '==',
            C8_tol = 1e-4
        ):
        '''Creates C8: A8 @ x <rel> b8
        Flexible constraint that constrains given crop areas and/or animal
        numbers in relation to given values.
        
        Parameters
        ----------
        C8_crp : pandas.Series
        C8_ani : pandas.Series
        C8_rel : string | '==', '>=' or '<='
        C8_tol : float

        Returns
        -------
        None
        '''

        self.A8 = self.make_A8(C8_crp, C8_ani)
        
        if (C8_crp is not None) & (C8_ani is not None):
            self.b8 = np.concatenate((C8_ani.values,C8_crp.values))
        elif C8_crp is not None:
            self.b8 = C8_crp.values
        else:
            self.b8 = C8_ani.values

        # Append code to include constraint when defining cvx problem
        if C8_rel == '==':
            self.cons_add_exec.extend([
                f'CONS.append(self.A8.M @ x >= self.b8 * {1-C8_tol})',
                f'CONS.append(self.A8.M @ x <= self.b8 * {1+C8_tol})'
            ])
        else:
            self.cons_add_exec.extend([
                f'CONS.append(self.A8.M @ x {C8_rel} self.b8)'
            ])

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

    def make_A3(self):

        # Get land uses to constrain
        land_uses = self.regions.par.get_unique(
            'land_use',
            qry='parameter == "max_land_use_factor"'
        )
        # Get row index from land uses and regions (lu,re)
        row_idx = pd.MultiIndex.from_product([
            land_uses,
            self.x_idx['crp'].get_level_values('region').unique()
        ], names=['land_use','region'])
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        # Get dict for translating crop --> land use
        rel = self.par.get_rel('crop','land_use')

        # Data and corresponding row/col numbers for constructing matrix
        val = [1 if rel[cr] == lu else 0 for lu in land_uses for cr,_,_ in col_idx]
        col_nr = list(range(len(col_idx))) * len(land_uses)
        row_nr = [row_idx.get_loc((lu,re)) for lu in land_uses for _,_,re in col_idx]

        M = scipy.sparse.coo_array((val,(row_nr,col_nr)), shape=(len(row_idx),len(col_idx))).tocsc()
        Z = scipy.sparse.csc_matrix((M.shape[0],len(self.x_idx['ani']))) # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z,M]),
            row_idx,
            {'ani':self.x_idx['ani'],'crp':col_idx}
        )

        return M
    
    def make_A5_1(self):

        # Get crop product and crop combindations where there is a constraint for maximum inclusion
        cps_crs = self.feed_mgmt.par.get_unique(['crop_prod','crop'], qry='parameter == "max_crop_in_crop_prod"')

        # Get row index (cp,cr,ps,re)
        row_idx = pd.MultiIndex.from_tuples([
            (cp,cr,ps,re)
            for cp,cr in cps_crs.values
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ], names = ['crop_prod','crop','prod_system','region'])
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

            # Check if herd has 'feed.max_supply_from_crop'
            if herd.feed.max_supply_from_crop is not None:
                # Go through production systems, crop products and crop combinations with a max supply constraint
                for ops,cp,cr in herd.feed.max_supply_from_crop.columns:
                    # Get maximum supply of crop product (cp) from output production system (ops) in region (re)
                    # from crop (cr) per head of defining animal of species (sp) and breed (br) in production
                    # system (ps), sub system (ss) and region (re)
                    res = - herd.feed.max_supply_from_crop.loc[:,(ops,cp,cr)]

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [row_idx.get_loc((cp,cr,ops,re)) for re in res.index]
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
    
    def make_A5_2(self):
        # Get crop product and crop combindations where there is a constraint for maximum inclusion
        cps_crs = self.feed_mgmt.par.get_unique(['crop_prod','crop'], qry='parameter == "max_crop_in_crop_prod"')

        # Get row index (cp,cr,ps,re)
        row_idx = pd.MultiIndex.from_tuples([
            (cp,cr,ps,re)
            for cp,cr in cps_crs.values
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ], names = ['crop_prod','crop','prod_system','region'])
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        row_idx_lookup = row_idx.droplevel('region').sort_values()

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for cp,cr,ps in row_idx.droplevel('region').unique():
            
            res = self.crops.production.loc[(cr,ps,slice(None)),(cp)].fillna(0)

            # Store values and row/col nr
            val.extend(res.values)
            row_nr.extend(
                [row_idx.get_loc((cp,cr,ps,re)) for re in res.index.get_level_values('region')]
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
        lu_rel = self.par.get_rel('crop','land_use')
        # Get dict for translating crop --> crop group
        cg_rel = self.par.get_rel('crop','crop_group')

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
    
    def make_A8(self, C8_crp, C8_ani):

        # Get row index (cr,ps,re), (sp,br,ps,ss,re)
        row_idx = {}
        # Get col index (cr,ps,re), (sp,br,ps,ss,re)
        col_idx = self.x_idx.copy()

        MS = []
        for ac in ['ani','crp']:
            if eval('C8_'+ac) is not None:
                row_idx[ac] = eval('C8_'+ac).index
                # Create identity matrix from col_idx
                n = len(col_idx[ac])
                M = scipy.sparse.identity(n, format='csc')
                # Drop rows to match row index
                sel_rows = [col_idx[ac].get_loc(i) for i in row_idx[ac]]
                M = M[sel_rows,:]
                # Create zero matrix and hstack
                if ac == 'ani':
                    Z = scipy.sparse.csc_matrix((M.shape[0],len(col_idx['crp'])))
                    MS.append(scipy.sparse.hstack([M,Z]))
                else:
                    Z = scipy.sparse.csc_matrix((M.shape[0],len(col_idx['ani'])))
                    MS.append(scipy.sparse.hstack([Z,M]))

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.vstack(MS),
            row_idx,
            col_idx
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
    
    def allocate_crop_production_per_use(self):
        '''Allocate crop areas to different uses.
        Creates attriute 'area_per_use' in CropProduction'''

        # Get prouction per crop product
        prod = (
            self.crops.production
            .stack()
            .groupby(['region','prod_system','crop_prod']).sum()
        )

        # Get crop product demand for feed per region
        feed_demand = (
            concat_herds(self.herds)
            .feed.crop_product_demand
            .xs('domestic', level='origin', axis=1)
            .groupby(['species','breed','sub_system','prod_system','crop_prod'], axis=1).sum()
            .stack(['prod_system','crop_prod'])
            .reindex(prod.index)
            .fillna(0)
        )
        feed_demand.columns = feed_demand.columns.map('feed ({0[0]}, {0[1]}, {0[2]})'.format).rename('demand')

        # Calculate feed demand met regionally as the maximum
        # share possible (i.e. regional crop areas are first
        # used to cater for regional feed demand befor national
        # demand for feed, food, etc.)
        regional_feed_demand = feed_demand.where(
            prod >= feed_demand.sum(axis=1),
            (
                feed_demand
                .mul(1/feed_demand.sum(axis=1), axis=0)
                .mul(prod, axis=0)
            )
        )

        prod_to_national = prod - regional_feed_demand.sum(axis=1)

        # Calculate remaining feed demand that needs to be supplied nationally
        national_feed_demand = (feed_demand - regional_feed_demand).groupby(['prod_system','crop_prod']).sum()

        national_demand = pd.concat([

            self.demand.crop_prod_demand,

            self.crops.seed_demand
            .groupby('prod_system')
            .sum().stack().rename('seed'),

            national_feed_demand

        ], axis=1).fillna(0).rename_axis('demand',axis=1)

        national_demand_shares = (
            national_demand
            .transform(lambda x: x/x.sum() if x.sum()>0 else 0, axis=1)
            .reindex(
                prod_to_national
                .reorder_levels(['prod_system','crop_prod','region'])
                .index
            )
            .reorder_levels(['region','prod_system','crop_prod'])
        )

        # Calculate total demand (regional+national)
        total_demand = (
            national_demand_shares.mul(prod_to_national, axis=0)
            +
            regional_feed_demand.reindex(national_demand_shares.columns, axis=1).fillna(0)
        )
        total_demand['none'] = prod - total_demand.sum(axis=1)
        # Set small negatives to zero
        assert total_demand.min().min() > -1e-6
        total_demand = total_demand.where(total_demand > 0, 0)

        # Calculate shares of total demand per use
        total_demand_shares = total_demand.mul(1/prod, axis=0)
        # Assume 100% none demand for rows with NaNs (i.e. where prod==0)
        total_demand_shares.loc[:,'none'].fillna(1, inplace=True)
        total_demand_shares.fillna(0, inplace=True)

        assert np.isclose(total_demand_shares.sum(axis=1),1).all()

        # Calculate crop production per use
        crop_production_per_use = multiply_aligned(

            total_demand_shares.unstack()
            .reindex(
                self.crops.production
                .reorder_levels(['region','prod_system','crop'])
                .index
            )
            .reorder_levels(['crop','prod_system','region']),

            self.crops.production
        ).groupby('demand', axis=1).sum()

        assert np.isclose(
            crop_production_per_use.sum(axis=1),
            self.crops.production.sum(axis=1)
        ).all()

        # Store dataframe as attribute
        self.crops.production_per_use = crop_production_per_use
        self.crops.data_attr.update(['production_per_use'])

class IndexedMatrix():
    '''Class to store pandas.Index/MultiIndex alongside a sparse
    matrix to keep track of things'''

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