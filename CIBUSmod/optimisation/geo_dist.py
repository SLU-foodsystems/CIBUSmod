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
    D : dict of pandas.Series
        A dict of pandas series representing demand for animal (D['ani']) and crop products (D['crp'])
        D['ani'].index should be on the form (production system, species, animal product)
        D['crp'].index should be on the form (production system, crop product)
    x0 : dict of pandas.Series
        A dict of pandas series representing emand for animald (D['ani']) and crop products (D['crp'])
        x0['ani'].index should be on the form (species, breed, production system, region)
        x0['crp'].index should be on the form (*land use*, *crop group* ,crop, production system, region)
    herds : pandas.Series of AnimalHerd objects
    crops : CropProduction object
        
    Attributes
    ----------'''

    def __init__(self,D,x0,crops,herds,feed_mgmt):
        
        self.D = D.copy()
        self.x0 = x0.copy()
        self.crops = crops
        self.herds = herds

        # Add rows for any domestically produced crop products used for feed not already in crop product demand vector (D['crp'])
        for cp in feed_mgmt.par.get_unique('crop_prod'):
            for ps in D['crp'].index.get_level_values('prod_system').unique():
                idx = (ps,cp)
                if (feed_mgmt.par.get('share_imported', crop_prod=cp, prod_system=ps) != 100).any() & (idx not in self.D['crp'].index):
                    self.D['crp'][idx] = 0

        # Define idex for x
        self.x_idx = {
            'ani' : pd.MultiIndex.from_tuples(
                    [(sp,br,ps,ss,re) for (sp,br,ps,ss) in self.herds.index for re in self.herds[(sp,br,ps,ss)].index],
                    names=['species','breed','prod_system','sub_system','region']
                    ),
            'crp' : self.crops.index
        }

        # Sort x0['ani'] to match x['ani']
        self.x0['ani'] = self.x0['ani'].loc[self.x_idx['ani'].droplevel('sub_system')]

        # Store D and x0 indexes
        self.D_idx = {
            'ani' : D['ani'].index,
            'crp' : D['crp'].index
        }
        
        self.x0_idx = {
            'ani' : self.x0['ani'].index,
            'crp' : self.x0['crp'].index
        }

        

    def make(self,use_cons='all',verbose=False):
        '''Creates constraints and defines optimisation problem'''

        vprint = verbose_init(verbose, id_str='GeoDist.make')

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
        if solver_settings=='default':
            solver_settings = {
                'solver':'OSQP',
                'max_iter':200000,
                'verbose':False
            }
        
        # If a list of alternative solver/settings is not supplied
        # make a one element list
        if not isinstance(solver_settings, list):
            solver_settings = [solver_settings]
        
        vprint('Finding solution ...')

        # Solver    Solution        Time (s)
        # ------    --------        --------
        # ECOS      infeasible      -
        # ECOS_BB   infeasible      -
        # OSQP      optimal         4.873e-02
        
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
            x = self.problem.variables()[0].value
            self.x = {
                'ani' : pd.Series(
                    x[0:len(self.x_idx['ani'])],
                    index = self.x_idx['ani']
                ),
                'crp' : pd.Series(
                    x[len(self.x_idx['ani']):],
                    index = self.x_idx['crp']
                )
            }
        else:
            vprint(f'No solution found!')
            self.success = False
            # NEED TO IMPLEMENT A WAY TO HANDLE THIS SITUATION

        vprint(type='end')


    def define_cvx_problem(self,use_cons):

        x = cvxpy.Variable(len(self.x_idx['ani'])+len(self.x_idx['crp']), nonneg=True)

        O1 = cvxpy.sum_squares(self.P1.M @ x - np.concatenate((self.x0['ani'].values,self.x0['crp'].values)))
        OBJ = cvxpy.Minimize(O1)

        CONS = []
        if '1' in use_cons:
            CONS.append(self.A1.M @ x == self.b1)
        if '2' in use_cons:
            CONS.append(self.A2.M @ x >= 0)
        if '3' in use_cons:
            CONS.append(self.A3.M @ x <= self.b3)
        if '4' in use_cons:
            CONS.append(self.A4.M @ x <= self.b4)
        if '5' in use_cons:
            pass
        if '6' in use_cons:
            pass
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

        # TAR FÖR LÅNG TID !!!

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
        '''Creates C3'''

        self.A3 = self.make_A3_and_A4(land_use='cropland')
        self.b3 = self.A3.M @ np.concatenate((self.x0['ani'],self.x0['crp'])) * 1.1

    def make_C4(self):
        '''Creates C4'''

        self.A4 = self.make_A3_and_A4(land_use='semi-natural grassland')
        self.b4 = self.A4.M @ np.concatenate((self.x0['ani'],self.x0['crp'])) * 1.1

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
            for ap in herd.production.columns.get_level_values('product').unique():
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

            # Go through crop products used as feed
            for cp in herd.feed.crop_product_demand.columns.get_level_values('crop_prod').unique():
                # Go through output production systems
                for ops in herd.feed.crop_product_demand.columns.get_level_values('prod_system').unique():
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
                    # Get production of crop product (cp) from production system (ps) per area of crop (cr)
                    # in production system (ps) and region (re)
                    res = self.crops.production.loc[(cr,ps,slice(None)),(cp)].fillna(0)

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
                                for cp in herd.feed.crop_product_demand['regional'].columns.get_level_values('crop_prod')
                                
                            ]))
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ])
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
                # Go through crop products with regional demand
                for cp in herd.feed.crop_product_demand['regional'].columns.get_level_values('crop_prod'):
                    # Go through output production systems
                    for ops in herd.feed.crop_product_demand['regional'].columns.get_level_values('prod_system'):
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
        reg_feeds = list(set([
                                cp
                                for herd in self.herds if 'regional' in herd.feed.crop_product_demand.columns
                                for cp in herd.feed.crop_product_demand['regional'].columns.get_level_values('crop_prod')
                                
                            ]))
        row_idx = pd.MultiIndex.from_tuples([
            (cp,ps,re)
            for cp in reg_feeds
            for ps in self.x_idx['ani'].get_level_values('prod_system').unique()
            for re in self.x_idx['ani'].get_level_values('region').unique()
        ],
        names = ['crop_prod','prod_system','region']
        )
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx['crp']

        row_idx_lookup = row_idx.droplevel('region').sort_values()

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []


        for cr,ps in self.crops.production[self.crops.production[reg_feeds].sum(axis=1)>0].index.droplevel('region').unique():
            for cp in row_idx.get_level_values('crop_prod'):
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

            # Get row index from animal product demand vector (re)
            row_idx = self.x_idx['crp'].get_level_values('region').unique()
            # Get col index from animal herds (cr,ps,re)
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



    # def make_A3(crs,pss,res):
    #     '''Creates A-matrix for C3 or C4:
    #     Arable area per region must not exceed available area
        
    #     C3: A3 @ x <= b3, where b3 is the available arable land per region
    #     C4: A4 @ x <= b4, where b4 is the available semi-natural grassland area per region'''
        
    #     r=0
    #     c=0
    #     data = []
    #     row_ind = []
    #     col_ind = []
        
    #     for re2 in res:
    #         c = 0
    #         for cr in crs:
    #             for ps in pss:
    #                 for re in res:
    #                     if (re == re2) & (cr != 'Semi-natural grasslands'):
    #                         row_ind.append(r)
    #                         col_ind.append(c)
    #                         data.append(1.0)
    #                     c+=1
    #         r+=1
            
    #     A = scipy.sparse.coo_array((data,(row_ind,col_ind)))
    #     return A.tocsc()

    # make_A4 = make_A3

    # def make_A5():
    #     pass

    # def make_A6(df,crs,pss,res):
    #     '''Creates A-matrix for C6 or C7:
        
    #     Constrain the maximum (C6) or minimum (C7) share of arable land devoted to a given crop in a given region in a
    #     given production system.
        
    #     C6(max): A6 @ x <= 0
    #     C7(min): A7 @ x >= 0
        
    #     Note to future:
    #     Currently implemented per crop but if one wants to constrain max/min share of a group of crops all crops
    #     corresponding to the group in a row should have (1-f) and the rest -1.
    #     Crops assumed not to be in rotation should have 0 in the matrix'''
        
    #     r=0
    #     c=0
    #     data = []
    #     row_ind = []
    #     col_ind = []
        
    #     crs2 = df.index.get_level_values('crop')[df['f']<1]
        
    #     if len(crs2)<1:
    #         return None
        
    #     for cr2 in crs2:
    #         f = df.loc[cr2,'f']
    #         for ps2 in pss:
    #             for re2 in res:
    #                 c = 0
    #                 for cr in crs:
    #                     for ps in pss:
    #                         for re in res:
    #                             if  (re==re2) & (ps==ps2) & (cr != 'Semi-natural grasslands'): # 
    #                                 if (cr==cr2):
    #                                     row_ind.append(r)
    #                                     col_ind.append(c)
    #                                     data.append(1-f)
    #                                 else:
    #                                     row_ind.append(r)
    #                                     col_ind.append(c)
    #                                     data.append(-f)

    #                             c+=1

    #                 r+=1
        
    #     A = scipy.sparse.coo_array((data,(row_ind,col_ind)))
    #     return A.tocsc()

    # make_A7 = make_A6

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