import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, fix_herds

if TYPE_CHECKING:
    from ..utils.retriever import ParameterRetriever
    from .feed_mgmt import FeedMgmt

class ManureMgmt():
    '''Management module that calculates animal manure excretion and losses.

    Parameters
    ----------
    herds : pd.Series of AnimalHerd objects
    feed_mgmt : FeedMgmt object
    par : ParameterRetriever object
    settings : dict
        Dict with <setting name> : <value>
        Allowed settings are:
            'MMS_grazing_from_feed' : bool, default False
                If True, use share of dry matter in ration
                from grazing to estimate the share of manure
                deposited on pasture
            'NPK_excretion_from_balance' : bool, default False
                If True, calculate N, P and K excretion from nutrients
                in feed minus nutrients incorporated in live weight gain
                and animal products. Only applies to AnimalHerds where
                the data attribute 'lwg' is calculated and relise on
                parameters '<N/P/K>_in_WL' and '<N/P/K>_in_prod'.
                If False, and for AnimalHerds where 'lwg' is not calculated
                N, P and K excretion is calculated from the parameter
                'manure_excr_<N/P/K>'
    '''

    def __init__(
            self,
            herds: pd.Series,
            feed_mgmt: "FeedMgmt",
            par: "ParameterRetriever",
            settings = {}
        ):

        self.par = par

        # Default settings
        self.settings = {
            'MMS_grazing_from_feed' : False,
            'NPK_excretion_from_balance' : False
        }
        # Update settings if valid input
        for k,v in settings.items():
            if k in self.settings:
                if type(v) is type(self.settings[k]):
                    self.settings.update({k:v})
                else:
                    raise TypeError(f'Expected {type(self.settings[k])} for setting "{k}"')
            else:
                raise ValueError(f'"{k}" is not a valid setting')

        self.herds = fix_herds(herds)
        self.feed_mgmt = feed_mgmt

        self.index = list(self.herds)[0].index

    def calculate(
            self,
            verbose=False
            ):
        '''Calculates all manure excretion and losses.

        Parameters
        ----------
        verbose : bool, default True
            Print progress messages

        Returns
        -------
        Nothing. Stores output in pandas.DataFrames in the attrubutes: '<element>_excr', '<element>_loss' and '<element>_to_spread'.
        <element> is VS (volatile solids), N (nitrogen), P (phosphorous) and K (potassium).
        '''

        # Define function to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='ManureMgmt')

        # Calculate manure management system (MMS) shares
        vprint('Calculating MMS shares ...')
        self.calculate_MMS_shares()

        # Calculate bedding material use
        vprint('Calculating bedding material use ...')
        self.calculate_bedding_material_use()

        # Volatile solids (VS)
        vprint('Calculating VS excretion ...')
        self.calculate_VS_excretion()
        vprint('Calculating VS losses ...')
        self.calculate_VS_losses()

        # Nitrogen (N), Phosphorous (P) and Potassioum (K)
        vprint('Calculating manure excretion ...')
        self.calculate_NPK_excretion(element = 'N')
        self.calculate_NPK_excretion(element = 'P')
        self.calculate_NPK_excretion(element = 'K')

        vprint('Calculating manure losses ...')
        self.calculate_NPK_losses(element = 'N')
        self.calculate_NPK_losses(element = 'P')
        self.calculate_NPK_losses(element = 'K')

        vprint(type='end')

    def calculate_MMS_shares(self):
        '''Calculate share of manure per manure management system (MMS). MMS = 'grazing' is calculated from the share of
        dry matter feed intake from grazing and the 'share_feed_on_pasture' parameter. Other MMS shares are defined with the
        MMS_share parameter'''

        # Clear ParameterRetriever filters
        self.par.clear()

        # Get defined manure management systems
        mmss = list(self.par.get_unique('MMS', qry="parameter == 'mms_share'"))

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            heads = herd.data_attr.get('heads')

            # Create dataframe
            mms_shares = pd.DataFrame(
                0.0,
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [tuple(list(i) + [mms]) for i in heads.columns for mms in mmss + ['grazing']],
                    names = heads.columns.names + ['MMS']
                )
            )

            # Get MMS shares (excl. grazing) according to parameters
            mms_shares.update(
                self.par.get_from_frame(
                    'mms_share',
                    mms_shares.loc[:,(slice(None), slice(None), mmss)]
                )/100
            )

            # Check that MMS shares add up to 100%, if not warn and correct
            if not np.isclose(mms_shares.T.groupby(['prod_system','animal']).sum().T, 1).all():
                warnings.warn(f'MMS shares did not add up to 100% for species: {herd.species}, breed: {herd.breed}. MMS shares were corrected.')
                mms_shares = mms_shares.T.groupby(['prod_system','animal']).transform(lambda x: x/x.sum()).T

            if self.settings['MMS_grazing_from_feed']:
                # Calculate share in MMS='grazing' from share of dry matter feed intake from grazing

                # Get feed intake
                feed_cons = herd.data_attr.get('feed.consumption')
                if 'grazing' in feed_cons.columns.get_level_values('feed'):

                    # Calculate share of feed intake from grazing
                    grazing_share = (
                        feed_cons.xs('grazing', level='feed', axis=1) /
                        feed_cons.T.groupby(['prod_system','animal']).sum().T.replace({0:np.nan})
                    )

                    # Add contribution of other feeds fed while on pasture
                    grazing_share += (
                        (100 - grazing_share) *
                        self.par.get_from_frame('share_feed_on_pasture', grazing_share)
                    )

                    # Add MMS column level
                    grazing_share = (
                        pd.concat({'grazing': grazing_share}, names=['MMS'], axis=1)
                        .reorder_levels(['prod_system','animal','MMS'], axis=1)
                    )

                    # Update DataFrame
                    mms_shares.update(grazing_share)

            else:
                # Calculate share in MMS='grazing' from 'grazing_period' parameter

                if 'grazing_period' in herd.par.data.index.get_level_values('parameter'):

                    # Clear and set filters
                    herd.par.clear()
                    herd.par.set(
                        species = herd.species,
                        breed = herd.breed,
                        sub_system = herd.sub_system
                    )

                    # Get grazing share
                    grazing_share = (
                        (herd.par.get_from_frame('grazing_period', herd.data_attr.get('heads')) / 12) *
                        (1 - herd.par.get_from_frame('indoors_during_grazing', herd.data_attr.get('heads'))/100)
                    )

                    # Add MMS column level
                    grazing_share = (
                        pd.concat({'grazing': grazing_share}, names=['MMS'], axis=1)
                        .reorder_levels(['prod_system','animal','MMS'], axis=1)
                    )

                    # Update DataFrame
                    mms_shares.update(grazing_share)

            if not np.isclose(mms_shares.T.groupby(['prod_system','animal']).sum().T, 1).all():
                # Adjust non-grazing MMS shares
                adjusted = (
                    mms_shares.loc[:, (slice(None), slice(None), mmss)]
                    .T.groupby(['prod_system','animal']).transform(lambda x: x/x.sum()).T
                )
                adjusted = multiply_aligned(
                    adjusted,
                    1 - mms_shares.xs('grazing', level='MMS', axis=1)
                )
                mms_shares.update(adjusted)

            # Drop MMSs not used in herd
            mms_sums = mms_shares.T.groupby('MMS').sum().T.sum()
            mms_shares = mms_shares.reindex(
                mms_sums.loc[mms_sums>0].index,
                level = 'MMS',
                axis = 1
            )

            herd.data_attr.add(
                mms_shares * 100,
                name = 'manure.mms_shares',
                unit = '%',
                orig = 'ManureMgmt',
                desc = 'Manure management systems shares',
                scalable = False
            )

        return None

    def calculate_bedding_material_use(self):
        '''Calculate use of bedding materials'''

        # Clear ParameterRetriever filters
        self.par.clear()
        self.feed_mgmt.par.clear()

        # Get types of bedding materials used
        fes = self.par.get_unique('feed', qry='parameter == "bedding_material_use"')

        for herd in self.herds:

            # Set species and breed filters
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Create dataframe
            df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [tuple(list(i) + [fe]) for i in herd.data_attr.get('manure.mms_shares').columns for fe in fes],
                    names = herd.data_attr.get('manure.mms_shares').columns.names + ['feed']
                )
            )

            # Calculate bedding material use [kg DM/year]
            DM = multiply_aligned(
                multiply_aligned(
                    self.par.get_from_frame('bedding_material_use', df),
                    herd.data_attr.get('manure.mms_shares')/100
                ) * 365.25,
                herd.data_attr.get('heads')
            )
            # Set bedding material use during grazing to zero
            if 'grazing' in DM.columns.get_level_values('MMS'):
                DM.loc[:,(slice(None),slice(None),['grazing'],slice(None))] *= 0

            # Calculate nitrogen,phosphorous and potassium in bedding materials
            N = ( # [kg N/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_N', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )
            P = ( # [kg P/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_P', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )
            K = ( # [kg K/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_K', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )

            # Add data attributes
            for df, element in zip([DM,N,P,K], ['DM','N','P','K']):
                herd.data_attr.add(
                    df,
                    name = f'bedding_material{"_"+element if not element=="DM" else ""}',
                    unit = f'kg {element}/year',
                    orig = 'ManureMgmt',
                    desc = 'Bedding material use' + ((' in terms of ' + element) if not element=='DM' else '')
                )

    def calculate_VS_excretion(self):
        '''Calculate VS excretion'''

        self.par.clear()
        pf = self.par.get_from_frame

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get production systems, animals in herd
            pss = herd.data_attr.get('heads').columns.get_level_values('prod_system').unique()
            anis = herd.animals

            DM_intake = herd.data_attr.get('feed.consumption').T.groupby(['prod_system','animal']).sum().T
            if herd.species != 'poultry':
                energy_manure = (
                    (
                        herd.data_attr.get('feed.ration_GE') -
                        herd.data_attr.get('feed.ration_DE') +
                        herd.data_attr.get('feed.ration_GE') * pf('UE_of_GE',herd.data_attr.get('feed.ration_GE'))
                    ) * DM_intake
                )
            else:
                energy_manure = (
                    (herd.data_attr.get('feed.ration_GE') - herd.data_attr.get('feed.ration_AME')) * DM_intake
                )

            VS_excr = (
                energy_manure *
                ((1 - herd.data_attr.get('feed.ration_ASH')/100) / herd.data_attr.get('feed.ration_GE'))
            ).fillna(0)

            # Distribute across MMS
            VS_excr = multiply_aligned(
                herd.data_attr.get('manure.mms_shares')/100,
                VS_excr
            )

            # ADD BEDDING TO VS EXCRETION!!!

            # Add data attribute
            herd.data_attr.add(
                VS_excr,
                name = 'manure.VS_excr',
                unit = 'kg VS/year',
                orig = 'ManureMgmt',
                desc = 'Manure volatile solids (VS) excretion'
            )

        return None

    def calculate_VS_losses(self):

        idx = pd.IndexSlice

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get excreted VS
            manure_VS = herd.data_attr.get('manure.VS_excr')

            # Separate manure destined for off-farm treatment
            # Assumes no VS loss on the farm for manure sent
            # for treatment
            manure_VS_to_treatment = \
                manure_VS * (self.par.get_from_frame('off-farm_treatment', manure_VS)/100)
            manure_VS = manure_VS - manure_VS_to_treatment

            # Calculate B0 for manure to treatment
            manure_B0_to_treatment = manure_VS_to_treatment * self.par.get_from_frame('methane_B0', manure_VS_to_treatment)

            # Compounds emitted
            css = ['CH4bio','CO2bio']

            # Create dataframe
            VS_loss = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms,cs) for ps, ani, mms in manure_VS.columns for cs in css],
                    names=['prod_system','animal','MMS','compound']
                ),
                dtype = float
            )

            # Calculate CH4 emissions using the IPCC Tier 2 method
            # from maximum methane production (B0) and MCF
            CH4_loss = (
                manure_VS *
                (self.par.get_from_frame('methane_B0', manure_VS)*0.67) *
                (self.par.get_from_frame('methane_MCF', manure_VS)/100)
            )

            # Calculate C and CO2 losses
            manure_C = manure_VS * (self.par.get_from_frame('manure_VS_C', manure_VS)/100)
            manure_C_to_treatment = manure_VS_to_treatment * (self.par.get_from_frame('manure_VS_C', manure_VS_to_treatment)/100)
            C_loss_tot = manure_C * (self.par.get_from_frame('C_loss', manure_C)/100)
            C_to_spread = manure_C - C_loss_tot

            C_loss_CH4 = CH4_loss * (12/(12+1*4))
            C_loss_CO2 = C_loss_tot - C_loss_CH4
            # Set negative CO2 loss to zero. Occurs if parameter 'C_loss' yields lower
            # C losses than calculated methane emissions (e.g. for MMS=grazing)
            C_loss_CO2 = C_loss_CO2.where(C_loss_CO2 > 0, 0.0)
            CO2_loss = C_loss_CO2 * ((12+16*2)/12)

            # Put results in dataframe
            VS_loss.loc[:,idx[:,:,:,'CH4bio']] = \
            CH4_loss.reindex(VS_loss.xs('CH4bio', level='compound', axis=1, drop_level=False).columns, axis=1)
            VS_loss.loc[:,idx[:,:,:,'CO2bio']] = \
            CO2_loss.reindex(VS_loss.xs('CO2bio', level='compound', axis=1, drop_level=False).columns, axis=1)

            # Add data attributes
            herd.data_attr.add(
                VS_loss,
                name = 'manure.VS_loss',
                unit = 'kg/year',
                orig = 'ManureMgmt',
                desc = 'Manure losses of C-containing compounds in stables and during manure storage'
            )
            herd.data_attr.add(
                C_to_spread,
                name = 'manure.C_to_spread',
                unit = 'kg C/year',
                orig = 'ManureMgmt',
                desc = 'Carbon (C) in manure available to spread'
            )
            herd.data_attr.add(
                manure_VS_to_treatment,
                name = 'manure.VS_to_treatment',
                unit = 'kg C/year',
                orig = 'ManureMgmt',
                desc = 'Volatile solids (VS) in manure to off-farm treatment'
            )
            herd.data_attr.add(
                manure_B0_to_treatment,
                name = 'manure.B0_to_treatment',
                unit = 'Nm3 CH4/year',
                orig = 'ManureMgmt',
                desc = 'Methane production potential (B0) in manure to off-farm treatment'
            )
            herd.data_attr.add(
                manure_C_to_treatment,
                name = 'manure.C_to_treatment',
                unit = 'kg C/year',
                orig = 'ManureMgmt',
                desc = 'Carbon (C) in manure to off-farm treatment'
            )

    def calculate_NPK_excretion(self, element):

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            if self.settings['NPK_excretion_from_balance'] and 'lwg' in herd.data_attr:
                # Calculate N excretion based on mass balance

                # Nutrient in feed input (excl. storage losses)
                # Motivation for not including storage losses:
                # Storage losses for roughages are to a large extent mass
                # losses during the ensilation process and for other feeds
                # storage losses are generally small.
                feed = (
                    (herd.data_attr.get('feed.consumption') + herd.data_attr.get('feed.feeding_losses'))
                    .T.groupby(['prod_system','animal']).sum().T *
                    (herd.data_attr.get('feed.ration_' + element) / 100)
                )

                # Nutrients in bedding materials
                bedding = herd.data_attr.get('bedding_material_' + element)

                # Nutrients in live weight gain
                lwg = (
                    herd.data_attr.get('lwg') *
                    self.par.get_from_frame(element + '_in_LW', herd.data_attr.get('lwg'))/1000
                )

                # Nutrients in products (excl. meat)
                prod = (
                    (
                        herd.data_attr.get('production') *
                        self.par.get_from_frame(
                            element + '_in_prod',
                            herd.data_attr.get('production')
                        )/1000
                    )
                    .T.groupby(['prod_system','animal']).sum().T
                )

                # Nutrients in milk fed to calves represent and export
                # from 'cows' and an input to 'calves, suckling'
                if 'milk_to_calves' in herd.data_attr:
                    # Get milk from cows to suckling calves
                    df = (
                        herd.data_attr.get('milk_to_calves')
                        .rename('milk')
                        .to_frame()
                        .rename_axis('animal_prod', axis=1)
                    )
                    # Calculate N/P/K content
                    df = (
                        df *
                        self.par.get_from_frame(element + '_in_prod', df)
                    ).loc[:,'milk']/1000
                    # Create DataFrame
                    milk_to_calves = pd.DataFrame(
                        0.0,
                        index = herd.index,
                        columns = feed.columns
                    )
                    # Insert N/P/K in milk to calves into the 'cows' columns as negative values
                    # and into the 'calves, suckling' column as positive values
                    milk_to_calves.loc[:,(herd.prod_system,'calves, suckling')] = df
                    milk_to_calves.loc[:,(herd.prod_system,'cows')] = -df
                else:
                    milk_to_calves = 0

                # Calculate excretion from animals
                excr_df = multiply_aligned(
                    herd.data_attr.get('manure.mms_shares')/100,
                    (feed - lwg - prod + milk_to_calves)
                )

                # Add nutrients in bedding materials
                excr_df = excr_df + bedding

            else:
                # Calculate N excretion from fixed factor per head
                excr_df = multiply_aligned(
                    (
                        herd.data_attr.get('manure.mms_shares')/100 *
                        self.par.get_from_frame('manure_excr_'+element, herd.data_attr.get('manure.mms_shares'))
                    ),
                    herd.data_attr.get('heads')
                )

            # Add data attribute
            herd.data_attr.add(
                excr_df,
                name = f'manure.{element}_excr',
                unit = f'kg {element}/year',
                orig = 'ManureMgmt',
                desc = f'Manure {_elem_to_name[element]} excretion'
            )

        return None

    def calculate_NPK_losses(self, element):

        # Get compounds emitted
        cmps = self.par.get_unique('compound', qry=f'f_element == "{element}"')

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system,
                element = element
            )

            # Create dataframe
            df = pd.DataFrame(
                0.0,
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [tuple(list(i) + [cmp]) for i in herd.data_attr.get('manure.mms_shares').columns for cmp in cmps],
                    names = herd.data_attr.get('manure.mms_shares').columns.names + ['compound']
                )
            )

            # Get excretion and propagate values to all compound columns
            excr = herd.data_attr.get(f'manure.{element}_excr')

            # Calculate losses in stables
            if (self.par.data.xs((element,'loss_stable'),level=('f_element','parameter'))>0).any():
                loss_factors_stable = self.par.get_from_frame('loss_stable', df)/100
                loss_stable = multiply_aligned(loss_factors_stable, excr)
            else:
                loss_stable = df * 0.0


            # Calculate remaining after stable losses
            to_storage = \
                excr - loss_stable.T.groupby(['prod_system','animal','MMS']).sum().T

            # Separate manure destined for off-farm treatment
            to_treatment = \
                to_storage * (self.par.get_from_frame('off-farm_treatment', to_storage)/100)
            to_storage = to_storage - to_treatment

            # Calculate losses in storage
            if (self.par.data.xs((element,'loss_storage'),level=('f_element','parameter'))>0).any():
                loss_factors_storage = self.par.get_from_frame('loss_storage', df)/100
                loss_storage = multiply_aligned(loss_factors_storage, to_storage)

                if element == 'N' and (self.par.data.xs(('N','loss_storage_of_TAN'),level=('f_element','parameter'))>0).any():
                    # Calculate storage losses expressed as share of TAN
                    loss_factors_storage_of_TAN = self.par.get_from_frame('loss_storage_of_TAN', df)/100
                    loss_storage_of_TAN = (
                        multiply_aligned(loss_factors_storage_of_TAN, to_storage) *
                        (self.par.get_from_frame('TAN_share', df)/100)
                    )

                    if ((loss_factors_storage>0) & (loss_factors_storage_of_TAN>0)).any().any():
                        warnings.warn('ManureMgmt: Storage losses for N expressed per total N and TAN for same item. Check data!')
                    
                loss_storage += loss_storage_of_TAN
            else:
                loss_storage = df * 0.0

            available_to_spread = \
                to_storage - loss_storage.T.groupby(['prod_system','animal','MMS']).sum().T


            # Add data attribute
            herd.data_attr.add(
                loss_stable + loss_storage,
                name = f'manure.{element}_loss',
                unit = f'kg {element}/year',
                orig = 'ManureMgmt',
                desc = f'Manure losses of {element}-containing compounds in stables and during manure storage'
            )
            herd.data_attr.add(
                    available_to_spread,
                    name = f'manure.{element}_to_spread',
                    unit = f'kg {element}/year',
                    orig = 'ManureMgmt',
                    desc = f'Total {_elem_to_name[element]} in manure available to spread after stable and storage losses'
            )
            herd.data_attr.add(
                    to_treatment,
                    name = f'manure.{element}_to_treatment',
                    unit = f'kg {element}/year',
                    orig = 'ManureMgmt',
                    desc = f'Total {_elem_to_name[element]} in manure to off-farm treatment'
            )

            if element=='N':
                # For nitrogen, also calculate and store plant available nitrogen available to spread
                # and going to off-farm treatment
                TAN_to_spread = (
                    available_to_spread *
                    (self.par.get_from_frame('TAN_share', available_to_spread)/100)
                ).T.groupby(['prod_system','animal','MMS']).sum().T
                TAN_to_treatment = (
                    to_treatment *
                    (self.par.get_from_frame('TAN_share', to_treatment)/100)
                ).T.groupby(['prod_system','animal','MMS']).sum().T

                herd.data_attr.add(
                    TAN_to_spread,
                    name = 'manure.TAN_to_spread',
                    unit = 'kg TAN/year',
                    orig = 'ManureMgmt',
                    desc = 'Total plant available nitrogen (TAN) in manure available to spread after stable and storage losses'
                )
                herd.data_attr.add(
                    TAN_to_treatment,
                    name = 'manure.TAN_to_treatment',
                    unit = 'kg TAN/year',
                    orig = 'ManureMgmt',
                    desc = 'Total plant available nitrogen (TAN) in manure to off-farm treatment'
                )

        return None

_elem_to_name = {
    'N' : 'nitrogen (N)',
    'P' : 'phosphorous (P)',
    'K' : 'potassium (P)'
}
