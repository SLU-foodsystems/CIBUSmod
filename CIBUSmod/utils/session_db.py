import os
import pandas as pd
import numpy as np

import json
from datetime import datetime, date
import sqlite3
import warnings
from contextlib import closing

from .retriever import ParameterRetriever
from ..main_modules.animal_herd import AnimalHerd, concat_herds

class Session(object):
    '''Class that handles the definition of scenarios and storing/retrieving
    output data from a database (SQLite) file.
    
    Parameters
    ----------
    name : str
        Session name
    data_path : str
        Path to data folder
    data_path_default : str (default None)
        Path to default data. If None 'data_path/default' is used
    data_path_scenarios : str (default None)
        Path to scenario data. If None 'data_path/scenarios' is used
    data_path_output : str (default None)
        Path to output data. If None 'data_path/output' is used
    timeout : float (default 5)
        How many seconds to wait for a locked table when
        trying to write to the database file
    max_cache : int (default 200)
        Number of database query results stored in cache for faster re-access
        Set to 0 to disable cache

    When a new Session object is initialised it checks if the file '<data_path>/output/<name>.sqlite' exists
    and if so connects to that database and any scenario definitions and output data within it.

    Printing the Session object will show information on defined scenarios and output data.

    Methods
    -------
    .add_scenario()              Defines a new scenario.
    .update_scenario()           Updates definition of existing scenario.
    .remove_scenario()           Removes a scenario (including any associated output data).
    .reorder_scenarios()         Changes the ordering of scenarios.
    .iterate()                   Used to iterate over scenarios and years.
    .store()                     Stores a model run in the output database file.
    .update_relation_tables()    Updates relation tables from .xlsx file
    .update_aggregation_rules()  Updates rules used to aggregate output data before storing.
    .clean()                     Defragments the database file to reduce filesize after dropping scenarios
    .get_attr()                  Get output data
    
    '''

    active_session = None # Stores the active Session instance

    def __init__(
            self,
            name,
            data_path,
            data_path_default = None,
            data_path_scenarios = None,
            data_path_output = None,
            timeout = 5,
            max_cache = 200
        ):
        
        self.name = name

        self.data_path = os.path.abspath(data_path)

        if data_path_default is not None:
            self.data_path_default = os.path.abspath(data_path_default)
        else:
            self.data_path_default = os.path.join(self.data_path, 'default')

        if data_path_scenarios is not None:
            self.data_path_scenarios = os.path.abspath(data_path_scenarios)
        else:
            self.data_path_scenarios = os.path.join(self.data_path, 'scenarios')

        if data_path_output is not None:
            self.data_path_output = os.path.abspath(data_path_output)
        else:
            self.data_path_output = os.path.join(self.data_path, 'output')

        self.db_path = os.path.join(self.data_path_output, self.name+'.sqlite')
        self.db_timeout = timeout

        # Create output folder if it does not already exist
        if not os.path.isdir(self.data_path_output):
            os.mkdir(self.data_path_output)

        # Dict that stores tables retrieved from database for faster access
        self.cache = CacheDict(max_size=max_cache)

        # Create main tables if they do not exist
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            con.execute("PRAGMA foreign_keys = 1;")

            res = cur.execute("""
                SELECT name FROM sqlite_schema
                WHERE 
                    type ='table' AND 
                    name NOT LIKE 'sqlite_%';
            """)
            tables_in_db = [r[0] for r in res.fetchall()]
            unexpected_tables = [t for t in tables_in_db if
                                 t not in ['scenarios', 'runs', 'data_attributes', 'relation_tables', 'aggregation_rules', 'levels'] and
                                 'attr_' not in t and
                                 'lvl_' not in t and
                                 'rel_' not in t]
            if unexpected_tables:
                raise ValueError(f"Unexpected tables found in '{self.db_path}': {unexpected_tables}")
                
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    scn_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    def TEXT
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY,
                    scn_id INTEGER,
                    year INTEGER,
                    calculated INTEGER DEFAULT 0,
                    
                    UNIQUE(scn_id, year),
                    
                    CONSTRAINT fk_scenarios
                        FOREIGN KEY (scn_id)
                        REFERENCES scenarios(scn_id)
                        ON DELETE CASCADE
                        ON UPDATE CASCADE
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_attributes (
                    attr_id INTEGER PRIMARY KEY,
                    module TEXT,
                    attr TEXT,
                    metadata TEXT,
                    
                    UNIQUE(module, attr)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS levels (
                    lvl_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS relation_tables (
                    rel_id INTEGER PRIMARY KEY,
                    lvl_name TEXT,
                    agr_name TEXT,

                    UNIQUE(lvl_name, agr_name)
                );
            """)

            cur.execute("""
                    CREATE TABLE IF NOT EXISTS aggregation_rules (
                        agg_id INTEGER PRIMARY KEY,
                        module TEXT,
                        attr TEXT,
                        agg_lvls TEXT,
    
                        UNIQUE(module, attr)
                    );
                """)

        self.update_relation_tables()
        
        if 'aggregation_rules' not in tables_in_db:

            # Default aggregations
            aggregation_rules = {
                ('CropProduction', 'fertiliser.manure_TAN') : ['species','animal_prod_system','MMS'],
                ('CropProduction', 'fertiliser.manure_N') : ['species','animal_prod_system','MMS'],
                ('CropProduction', 'fertiliser.manure_P') : ['species','animal_prod_system','MMS'],
                ('CropProduction', 'fertiliser.manure_K') : ['species','animal_prod_system','MMS'],
                ('CropProduction', 'fertiliser.manure_C') : ['species','animal_prod_system','MMS'],
                ('CropProduction', 'fertiliser.manure_N_soil_loss') : ['species','compound'],
                ('CropProduction', 'fertiliser.manure_N_application_loss') : ['species','compound']
            }
                
            self.update_aggregation_rules(aggregation_rules)
        
        self.activate()
        
    def _repr_html_(self):

        file_size = os.path.getsize(self.db_path)
        for ext in ['bytes','KB','MB','GB','TB']:
            if file_size < 1024 or ext == 'TB':
                size_str = f"{file_size:.1f} {ext}"
                break
            else:
                file_size /= 1024

        
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            scns = pd.read_sql_query("SELECT * FROM scenarios ORDER BY scn_id", con, index_col='scn_id')
            runs = pd.read_sql_query("SELECT * FROM runs ORDER BY scn_id, year", con, index_col=['scn_id','year'])
            data_attrs = pd.read_sql_query("SELECT * FROM data_attributes", con, index_col=['module', 'attr_id'])
            levels = pd.read_sql_query("SELECT * FROM levels", con, index_col='lvl_id')['name']

            # Get levels for each data attribute
            data_attrs['levels'] = None
            for attr_id in data_attrs.index.unique('attr_id'):
                res = cur.execute(f"PRAGMA table_info(attr_{attr_id})")
                data_attrs.loc[(slice(None), attr_id), 'levels'] = \
                    str([levels.loc[int(c[1].replace('lvl_',''))] for c in res.fetchall() if 'lvl' in c[1]])

        # Create basic information html
        main_html = f"""
        <h3>CIBUSmod Session</h3>
        <div>
            <b>Name:</b> {self.name}<br>
            <b>Path:</b> {self.db_path}<br>
            <b>File size:</b> {size_str}<br>
            <b>Active:</b> {'Yes' if self is self.active_session else 'No'}
        </div>
        """

        # Create scenarios information html
        scns_html = """
        <h4>Scenarios</h4>
        """
        if len(scns) > 0:
            for scn_id in scns.index:
                name = scns.loc[scn_id]['name']
                scn_def = json.loads(scns.loc[scn_id]['def'])
                if 'scenario' in scn_def:
                    scn_def['scenario_workbooks'] = scn_def.pop('scenario')
                    
                scns_html += f"""
        <details>
            <summary style="font-size: 14px;">{name}</summary>
            <div style="padding-left: 20px;">
                <b>scenario_workbooks:</b> {scn_def['scenario_workbooks']}<br>
                <b>modules:</b> {scn_def['modules']}<br>
                <b>pars:</b> {scn_def['pars']}
            </div>
            <div style="padding-left: 20px;">
                {runs.xs(scn_id)[['calculated']].replace({0:'Not calculated'}).to_html()}
            </div>
        </details>
                """
        else:
            scns_html += """
        <div>
            No scenarios defined
        </div>
            """

        # Create data attributes information html
        data_attrs_html = """
        <h4>Data attributes</h4>
        """
        if len(data_attrs) > 0:
            for mod in data_attrs.index.unique('module'):
                data_attrs_mod = data_attrs.xs(mod).sort_values('attr')
                data_attrs_html += f"""
        <details>
            <summary style="font-size: 14px;">{mod}</summary>
            <div style="padding-left: 20px;">
                """
                for attr_id in data_attrs_mod.index:
                    attr = data_attrs_mod.loc[attr_id, 'attr']
                    metadata = json.loads(data_attrs_mod.loc[attr_id, 'metadata'])
                    levels = data_attrs_mod.loc[attr_id, 'levels']
                    data_attrs_html += f"""
                    <details>
                        <summary>{attr}</summary>
                        <div style="padding-left: 20px;">
                            <b>Description:</b> {metadata['desc']}<br>
                            <b>Unit:</b> {metadata['unit']}<br>
                            <b>Origin module:</b> {metadata['orig']}<br>
                            <b>Levels:</b> {levels}
                        </div>
                    </details>
                    """
                data_attrs_html += """
            </div>
        </details>
                """
        else:
            data_attrs_html += """
        <div>
            No data stored
        </div>
            """

        # Combine parts
        html = main_html + scns_html + data_attrs_html + """
        <div style="padding-top: 20px;">
            <i><b>Note:</b> Items under Scenarios and Data attributes are clickable to show more details</i>
        </div>
        """
        
        return html

    def __getitem__(self, scn):

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
                
            try:
                res = cur.execute(f"""
                    SELECT def FROM scenarios WHERE name == "{scn}"
                """)
            except:
                raise KeyError(f"No scenario named '{scn}'")   

            scn_def = json.loads(res.fetchone()[0])

            # For compatibility with older database files
            if 'scenario' in scn_def:
                scn_def['scenario_workbooks'] = scn_def.pop('scenario')
        
        return scn_def
    
    def activate(self):
        """Activates the Session instance and sets ParameterRetriever data folders"""

        self.__class__.active_session = self

        ParameterRetriever.set_data_folder(
            path=self.data_path,
            default_path=self.data_path_default,
            scenarios_path=self.data_path_scenarios
        )

        self.cache = CacheDict()

        return None
    
    def scenarios(self, subset='all'):
        """Get defined scenarios

        Parameters
        ----------
        subset : str
            Scenarios/years to return; 'all', 'has output' or 'no output'

        Returns
        -------
        Dict with scenario names as keys and list of years as values
        """

        qry = """
            SELECT name, year
                FROM runs AS r
            INNER JOIN scenarios AS s
                ON r.scn_id == s.scn_id
        """

        if subset == 'no output':
            qry += """
                WHERE calculated == 0
            """
        elif subset == 'has output':
            qry += """
                WHERE calculated != 0
            """

        qry += """
            ORDER BY r.scn_id, year
        """

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
            scenarios = {}
            res = cur.execute(qry)
            for r in res:
                if r[0] not in scenarios:
                    scenarios[r[0]] = [str(r[1])]
                else:
                    scenarios[r[0]] += [str(r[1])]

        return scenarios

    def iterate(self, subset='no output'):
        '''Iterates over scenarios and years
        
        Parameters
        ----------
        subset : 'all', 'no output' (default), date/time string or (list of) scenario name(s)
            If 'all', all scenarios are iterated over
            If 'no output', only scenarios without outputs are iterated over
            If (list of) scenario name(s) are provided those scenarios are iterated over
        
        Example:
        for scn, year in obj.iterate():
        
            <-- CIBUSmod run code -->
            
            obj.store(
                scn = scn,
                year = year,
                ...
            )
            
        '''
        
        qry = """
            SELECT name, year
                FROM runs AS r
            INNER JOIN scenarios AS s
                ON r.scn_id == s.scn_id
        """
        
        if subset == 'no output':
            qry += """
                WHERE calculated == 0
            """
        elif subset != 'all':
            if isinstance(subset, str):
                try:
                    # Try to parse as date/time string
                    cutoff_date = _parse_datetime_str(subset)
                    qry += f"""
                        WHERE datetime(calculated) < datetime("{cutoff_date}")
                    """
                except ValueError:
                    # If that fails see if it's a scenario name
                    if subset in self.scenarios():
                        qry += f"""
                            WHERE name == "{subset}"
                        """
                    else:
                        raise ValueError('subset string not recognized as date/time nor a scenario name')
            else:
                qry += f"""
                    WHERE name IN {tuple(subset)}
                """
                
        qry += """
            ORDER BY r.scn_id, year
        """
        
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
        
            res = cur.execute(qry)
            res = res.fetchall()
        
        for row in res:
            scn = row[0]
            year = row[1]
            yield (scn, str(year))

    def add_scenario(self, name, years, scenario_workbooks=None, modules='all', pars='all'):
        '''Adds a scenario to the Session object

        Parameters
        ----------
        name : str
            Name of scenario (not necessarily the same as the name of the scenario Excel sheet)
        years : (list of) int or str
            Years to be run
        scenario_workbooks : None or (list of) str, optional
            Name of scenaro Excel workbook(s) to use. If None default data are used
            If a list is supplied data is uppdated based on all scenario workbooks but if
            the same parameter is updated in several workbooks only the latest one in the
            list will have an effect.
        modules : 'all' or (list of) str with module names, optional
            Modules to be updated. If 'all', all modules will be updated otherwise only the
            modules specified will be updated according to the scenario Excel file(s)
        pars : 'all' or (list of) str with parameter names
                or dict with module:[parameter(s)], optional
            Parameters to be updated. If 'all', all parameters are updated otherwise only the
            specified parameters are updated (see examples)

            Example 1:
            pars = ['par_A', 'par_B']
            This will only update 'par_A' and 'par_B' across all modules

            Example 2:
            pars = {'Mod1' : ['par_A', 'par_B'], 'Mod2' : 'par_C'}
            This will only update 'par_A' and 'par_B' in module 'Mod1', only update 'par_C'
            in 'Mod2' and all parameters in any other module.
        
        '''
    
        if not isinstance(years, list):
            years = [years]

        # Crate scenario definition dict
        scn_def = {
            'scenario_workbooks' : scenario_workbooks,
            'modules' : modules,
            'pars' : pars,
        }
        scn_def_json = json.dumps(scn_def)
      
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
            
            # Insert scenario into database if it's not alredy there
            try:
                cur.execute("""
                    INSERT INTO scenarios(name, def) VALUES(?, ?);
                """, (name, scn_def_json))
            except sqlite3.IntegrityError:
                print(f"A scenario with the name '{name}' already exists use .update_scenario() or .remove_scenario() instead.")
                return None

            res = cur.execute(f"""
                SELECT scn_id FROM scenarios WHERE name == "{name}"
            """)
            scn_id = res.fetchone()[0]

            data = [(scn_id, y) for y in years]
            cur.executemany("""
                INSERT INTO runs(scn_id, year) VALUES(?, ?);
            """, data)
                
        return None

    def update_scenario(self, name, years=None, scenario=None, modules=None, pars=None, prompt=True):
        '''Updates a scenario in the Session object

        Parameters
        ----------
        name : str
            Name of scenario (not necessarily the same as the name of the scenario Excel sheet)
        years : (list of) int or str, optional
            Years to be run
        scenario : None or (list of) str with scenario Excel sheet name(s), optional
            Name of scenaro Excel sheet(s) to use. If None default data are used
            If a list is supplied data is uppdated based on all scenario Excel sheets but if
            the same parameter is updated in several Excel sheets only the latest one in the
            list will have an effect.
        modules : 'all' or (list of) str with module names, optional
            Modules to be updated. If 'all', all modules will be updated otherwise only the
            modules specified will be updated according to the scenario Excel file(s)
        pars : 'all' or (list of) str with parameter names
                or dict with module:[parameter(s)], optional
            Parameters to be updated. If 'all', all parameters are updated otherwise only the
            specified parameters are updated (see examples)
        prompt : bool, default True
            If True, require confirmation before deleting any data

            Example 1:
            pars = ['par_A', 'par_B']
            This will only update 'par_A' and 'par_B' across all modules

            Example 2:
            pars = {'Mod1' : ['par_A', 'par_B'], 'Mod2' : 'par_C'}
            This will only update 'par_A' and 'par_B' in module 'Mod1', only update 'par_C'
            in 'Mod2' and all parameters in any other module.
        
        '''

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            con.execute("PRAGMA foreign_keys = 1;")

            res = cur.execute("""
                SELECT scn_id, def FROM scenarios
                    WHERE name = ?
            """, (name,))
            res = res.fetchone()
                
            if res is None:
                raise KeyError(f"No scenario with the name '{name}'")

            scn_id = res[0]
            old_def = json.loads(res[1])

            res = cur.execute("""
                SELECT year, calculated FROM runs
                    WHERE scn_id = ?
            """, (scn_id,))
            res = res.fetchall()
            old_years = [r[0] for r in res]
            old_years_w_data = [r[0] for r in res if r[1] != 0]

            new_def = old_def.copy()

            if scenario is not None:
                new_def['scenario'] = scenario
            if modules is not None:
                new_def['modules'] = modules
            if pars is not None:
                new_def['pars'] = pars
                
            if years is not None:
                if not isinstance(years, list):
                    new_years = [years]
                else:
                    new_years = years
                new_years = [int(y) for y in new_years]
            else:
                new_years = old_years.copy()

            if new_def != old_def:
                if old_years_w_data and prompt:
                    ui = input('This will remove all output data associated with this scenario. Proceed? (Y/N)')
                    if ui.capitalize() != 'Y':
                        return None

                cur.execute("""
                    UPDATE scenarios
                    SET
                        def = ?
                    WHERE
                        scn_id = ?
                """, (json.dumps(new_def), scn_id))
                
                cur.execute("""
                    DELETE FROM runs WHERE scn_id = ?
                """, (scn_id,))

                con.commit()
                
                data = [(scn_id, y) for y in new_years]
                cur.executemany("""
                    INSERT INTO runs(scn_id, year) VALUES(?, ?);
                """, data)
                
            else:
                years_to_add = [y for y in new_years if y not in old_years]
                years_to_drop = [y for y in old_years if y not in new_years]
                if any([y in years_to_drop for y in old_years_w_data]) and prompt:
                    ui = input('Some years with calculated output data will be removed. Proceed? (Y/N)')
                    if ui.capitalize() != 'Y':
                        return None
                        
                data = [(scn_id, y) for y in years_to_drop]
                cur.executemany("""
                    DELETE FROM runs
                        WHERE scn_id = ? AND year = ?;
                """, data)

                con.commit()

                data = [(scn_id, y) for y in years_to_add]
                cur.executemany("""
                    INSERT INTO runs(scn_id, year) VALUES(?, ?);
                """, data)

        self.cache.clear()
        
        return None
                    

    def remove_scenario(self, name, prompt=True):
        '''Removes named scenario including all output data
        
        Parameters
        ----------
        name : str
            Name of scenario to be removed
        prompt : bool, default True
            If True, require confirmation before deleting any data

        Returns
        -------
        None
        
        '''

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            con.execute("PRAGMA foreign_keys = 1;")

            res = cur.execute("""
                SELECT calculated
                    FROM runs
                INNER JOIN scenarios
                    ON runs.scn_id = scenarios.scn_id
                WHERE name = ?
            """, (name,))

            # Require user confirmation if scenario has data
            if any([r[0] for r in res.fetchall()]) and prompt:
                ui = input('This scenario has output data that will also be removed. Proceed? (Y/N)')
                if ui.capitalize() != 'Y':
                    return None
        
            cur.execute("""
                DELETE FROM scenarios WHERE name = ?
            """, (name,))

        self.cache.clear()

        return None

    def reorder_scenarios(self, names):
        """Changes the order of scenarios
        
        Parameters
        ----------
        names : list of str
            List of all scenario names in the desired order

        Returns
        -------
        None
        """

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            con.execute("PRAGMA foreign_keys = 1;")

            res = cur.execute("""
                SELECT scn_id, name FROM scenarios ORDER BY scn_id
            """)
            res = res.fetchall()
                
            old_ids = [r[0] for r in res]
            old_names = [r[1] for r in res]

            if set(old_names) != set(names):
                raise ValueError(f'names must be a list of all scenario names in Session: {old_names}')

            temp_ids = list(range(max(old_ids)+1, len(names)+max(old_ids)+1)) # temp. scn_ids to avoid violating UNIQUE constraint
            new_ids = list(range(1, len(names)+1))
                
            temp_ids_names = [(tid,n) for tid,n in zip(temp_ids,names)]
            new_ids_names = [(id,n) for id,n in zip(new_ids,names)]

            cur.executemany("""
                UPDATE scenarios
                SET scn_id = ?
                WHERE name = ?
            """, temp_ids_names)

            cur.executemany("""
                UPDATE scenarios
                SET scn_id = ?
                WHERE name = ?
            """, new_ids_names)

            self.cache.clear()

        return None

    def clean(self):
        """Performes VACUUM; on database file"""

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con:
            con.execute("VACUUM;")
        

    def store(self, scn, year, *args):
        '''Write output data to database file.
        
        Parameters
        ----------
        scn : str
            Scenario name
        year : str
            Year
        *args : CIBUSmod main moduels / iterable of AnimalHerd objects
            Pass in the modules to store output data for
            
        '''

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
            
            res = cur.execute("""
                SELECT run_id
                    FROM runs
                INNER JOIN scenarios
                    ON runs.scn_id = scenarios.scn_id
                WHERE name = ? AND year = ?
            """, (scn, year))

            try:
                run_id = res.fetchone()[0]
            except TypeError:
                raise ValueError(f'Scenario "{scn}" and year {year} not defined')

            # Get aggregation rules
            res = cur.execute("""
                SELECT module, attr, agg_lvls
                FROM aggregation_rules
            """)
            aggregation_rules = {(i[0], i[1]) : json.loads(i[2]) for i in res.fetchall()}
        
        print(f"Writing outputs to '{self.db_path}'")
       
        for arg in args:

            if _isiterable(arg):
                if len(arg)>0 and np.all([isinstance(h,AnimalHerd) for h in arg]):
                    arg = concat_herds(arg)
                else:
                    warnings.warn('Passed iterable of non-AnimalHerd objects ignored')
                    continue

            if hasattr(arg, 'module_name') and hasattr(arg, 'data_attr'):

                module = arg.module_name
                
                # Only include 'scalable' data attributes (i.e. those that can be aggregated)
                data_attr_dict = {k : v for k, v in arg.data_attr.items() if v['scalable']}

                with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
                    closing(con.cursor()) as cur:

                    data = [(module, attr, json.dumps(meta)) for attr, meta in data_attr_dict.items()]
                    
                    cur.executemany("""
                        INSERT OR IGNORE
                            INTO data_attributes(module, attr, metadata)
                            VALUES(?,?,?);
                    """, data)

                for attr in data_attr_dict:
                    print(attr)

                    data_to_write = _get_check_and_clean_data(arg, module, attr)

                    # Aggregate data according to aggregation rules
                    if (module, attr) in aggregation_rules:
                        if isinstance(data_to_write, pd.DataFrame):
                            agg_lvls = aggregation_rules[(module, attr)]
                            agg_lvls = [lvl for lvl in agg_lvls if lvl in data_to_write.columns.names]
                            data_to_write = data_to_write.T.groupby(agg_lvls).sum().T


                    with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
                        closing(con.cursor()) as cur:
    
                        res = cur.execute("""
                            SELECT attr_id FROM data_attributes
                                WHERE module=? AND attr=?;
                        """, (module, attr))
                        attr_id = res.fetchone()[0]
                    
                    if isinstance(data_to_write, pd.DataFrame|pd.Series):
                        data_to_write = _level_names_to_integer_key(data_to_write, self.db_path, self.db_timeout)
                        
                        lvl_cols = data_to_write.index.names
                        data = pd.concat(
                            {run_id: data_to_write},
                            names=['run_id']
                        ).rename('value').to_frame()
                    else:
                        lvl_cols = []
                        data = pd.DataFrame(
                            [(run_id,data_to_write)],
                            columns=['run_id','value']
                        ).set_index('run_id')

                    # Table creation SQL query
                    create_qry = f"""
                        CREATE TABLE IF NOT EXISTS attr_{attr_id} (
                            run_id INTEGER,"""
                    for lvl_col in lvl_cols:
                        create_qry += f"""
                            {lvl_col} INTEGER NOT NULL,"""
                    create_qry += """
                            value NUMERIC,
                            """
                    create_qry += f"""
                            PRIMARY KEY({', '.join(['run_id']+lvl_cols)}),
                    """

                    create_qry += """
                            CONSTRAINT fk_runs
                                FOREIGN KEY (run_id)
                                REFERENCES runs(run_id)
                                ON DELETE CASCADE
                                ON UPDATE CASCADE
                        ) WITHOUT ROWID;
                    """

                    # Row deletion SQL query
                    delete_qry = f"""
                        DELETE FROM attr_{attr_id} WHERE run_id = ?
                    """

                    with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
                        closing(con.cursor()) as cur:

                        con.execute("PRAGMA foreign_keys = 1;")

                        # Create table if it does not exists
                        cur.execute(create_qry)

                        # Drop any existing data with the same run_id
                        cur.execute(delete_qry, (run_id,))
                            
                        # Insert data
                        data.to_sql(
                            name = f"attr_{attr_id}",
                            con = con,
                            if_exists = 'append',
                            index = True,
                            method = None # 'multi' does not work
                        )
                   
            else:
                warnings.warn(f'Passed object of type {type(arg)} ignored')

        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cur.execute("""
                UPDATE runs
                    SET calculated = ?
                WHERE run_id = ?
            """, (timestamp, run_id))

        self.cache.clear()
        
        print("Outputs stored!")

        return None

    def update_relation_tables(self):
        """Updates relation tables in SQLite database from relation_tables.xlsx
        """
        
        path = os.path.join(self.data_path,'relation_tables.xlsx')

        try:
            xl = pd.ExcelFile(path)
        except FileNotFoundError:
            warnings.warn(f"Could not update relation tables. '{str(path)}' not found.")
            return None

        for sheet in xl.sheet_names:
            lvl_name = sheet
            df = (
                xl.parse(sheet)
                .set_index(lvl_name)
            )
            for agr_name in df.columns:
                
                with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
                    closing(con.cursor()) as cur:
        
                    cur.execute("""
                        INSERT OR IGNORE
                            INTO relation_tables(lvl_name, agr_name)
                            VALUES(?,?)
                    """, (lvl_name, agr_name))
        
                    res = cur.execute("""
                        SELECT rel_id FROM relation_tables
                            WHERE lvl_name = ? AND agr_name = ?
                    """, (lvl_name, agr_name))
                    rel_id = res.fetchone()[0]

                    df_to_write = (
                        df
                        [agr_name]
                        .reset_index()
                        .rename(columns = {agr_name:'agr_value', lvl_name:'lvl_value'})
                    )
        
                    df_to_write.to_sql(
                        name = f"rel_{rel_id}",
                        con = con,
                        if_exists = 'replace',
                        index = False
                    )

        self.cache.clear()
        
        return None

    def update_aggregation_rules(self, aggregation_rules, prompt=True):
        """Update aggregation rules

        Parameters
        ----------
        aggregation_rules : dict
            The dict should have a tuple with (<module name>, <attribute name>) as keys
            and a (list of) str designating column levels to aggregate data to.
        prompt : bool, default True
            If True, require confirmation before deleting any data
        """
        
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
    
            res = cur.execute("""
                SELECT module, attr, agg_lvls
                FROM aggregation_rules
            """)
            aggregation_rules_old = {(i[0], i[1]) : json.loads(i[2]) for i in res.fetchall()}
    
            res = cur.execute("""
                SELECT module, attr
                FROM data_attributes
            """)
            attr_w_data = res.fetchall()
    
            to_write = []
            to_delete = []
            to_delete_data = []
            for i in aggregation_rules:
                if i in aggregation_rules_old:
                    if aggregation_rules[i] == aggregation_rules_old[i]:
                        continue
                    elif aggregation_rules[i] is None:
                        to_delete += [i]
                    elif isinstance(aggregation_rules[i], list|str):
                        to_write += [i]
                        to_delete += [i]
                    else:
                        raise ValueError('Aggregation levels must be supplied as a str or list of str')
                else:
                    if aggregation_rules[i] is None:
                        continue
                    elif isinstance(aggregation_rules[i], list|str):
                        to_write += [i]
                    else:
                        raise ValueError('Aggregation levels must be supplied as a str or list of str')
                        
                if i in attr_w_data:
                    to_delete_data += [i]
    
            # Drop affected data attribute tables
            if attrs := ['.'.join(i) for i in to_delete_data]:
                if prompt:
                    ui = input(f'This will delete output data: {attrs}. Are you sure? (Y,N)')
                else:
                    ui = 'Y'
                if ui.capitalize() == 'Y':
                    for mod_attr in to_delete_data:
                        res = cur.execute("""
                            SELECT attr_id FROM data_attributes
                            WHERE module = ? AND attr = ?
                        """, mod_attr)
                        attr_id = res.fetchone()[0]
        
                        cur.execute(f"""
                            DROP TABLE IF EXISTS attr_{attr_id}
                        """)
                        cur.execute("""
                            DELETE FROM data_attributes
                            WHERE attr_id = ?
                        """, (attr_id,))
                else:
                    return
    
            # Delete aggregation rules
            for mod_attr in to_delete:
                cur.execute("""
                    DELETE FROM aggregation_rules
                    WHERE module = ? AND attr = ?
                """, mod_attr)
    
            # Insert aggregation rules
            for mod_attr in to_write:
                cur.execute("""
                    INSERT INTO aggregation_rules(module, attr, agg_lvls)
                    VALUES(?,?,?)
                """, mod_attr + (json.dumps(aggregation_rules[mod_attr]),))

    def get_attr(
            self,
            module,
            attr,
            groupby = 'all',
            scn = 'all',
            years = 'all',
            all_region_levels = True,
            interpolate = False
        ):
    
        '''Get specified data attribute from output.
                
        Parameters
        ----------
        module : str
            Module to get output from: 'DemandAndConversions', 'Regions', 'CropProduction' or 'AnimalHerd'
        attr : str
            data attribute to get
        groupby : str, list or dict, default 'all'
            If str or list data is grouped and aggregated by these levels.
            If 'all' all available levels are returned
            If 'none'  data is summed over all levels
            If a dict is supplied relation tables are used
        scn : (list of) str
        years : (list of) str
        all_region_levels : Bool, default True
            If True, returns with a column index containing all regions
            (filling with zeros)
        interpolate : Bool, default False
            If True interpolate between defined years
            
        Returns
        -------
        pandas.DataFrame or Series with scenario (scn) and year as index and <groupby>
        as columns.
        '''
        
        with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
            closing(con.cursor()) as cur:
      
            res = cur.execute(f"""
                SELECT attr_id, module, attr FROM data_attributes
                    WHERE module LIKE '{module}%' AND attr LIKE '{attr}%'
                    ORDER BY length(module), length(attr)
                """)
            all_res = res.fetchall()
            if len(all_res) == 0:
                msg = f"No match for module='{module}' and attr='{attr}'"
                raise ValueError(msg)
            elif len(all_res) > 1:
                exact = [r for r in all_res
                         if (r[2].capitalize() == attr.capitalize() and module.capitalize() in r[1].capitalize())
                         or (attr.capitalize() in r[2].capitalize() and r[1].capitalize() == module.capitalize())]
                if len(exact) > 1:
                    exact = [r for r in exact
                            if (r[2].capitalize() == attr.capitalize() and r[1].capitalize() == module.capitalize())]
                if len(exact) == 1:
                    attr_id = exact[0][0]
                    module = exact[0][1]
                    attr = exact[0][2]
                else:
                    msg = f"Multiple matches for module='{module}' and attr='{attr}':\n" + '\n'.join([str(row) for row in all_res])
                    raise ValueError(msg)
            else:
                attr_id = all_res[0][0]
                module = all_res[0][1]
                attr = all_res[0][2]

            if groupby == 'all':
                # Get all levels in table
                res = cur.execute(f"SELECT * FROM attr_{attr_id}")
                lvl_ids_in_table = [int(t[0].split('_')[1]) for t in res.description if 'lvl_' in t[0]]
                groupby = []
                for lvl_id in lvl_ids_in_table:
                    res = cur.execute("""
                        SELECT name FROM levels
                            WHERE lvl_id = ?
                    """, (lvl_id,))
                    groupby += [res.fetchone()[0]]
            if groupby == 'none':
                groupby = []
            if isinstance(groupby, str):
                groupby = [groupby]
            if isinstance(groupby, list):
                groupby = {k:None for k in groupby}
        
            # Construct SQL query
            qry_lvls_sel = []
            qry_lvls_join = []
            qry_lvls_group = []
                
            lvl_ids = []
            for lvl in groupby:
                res = cur.execute("""
                    SELECT lvl_id FROM levels
                        WHERE name = ?
                """, (lvl, ))
                try:
                    lvl_id = res.fetchone()[0]
                except TypeError:
                    raise ValueError(f"'{lvl}' not a valid level name")
        
                lvl_ids += [lvl_id]
                
                qry_lvls_join += [f"LEFT JOIN lvl_{lvl_id} ON lvl_{lvl_id} = lvl_{lvl_id}.lvl_value_id"]
        
                agr_names = groupby[lvl]
                if not isinstance(agr_names, list):
                    agr_names = [agr_names]
        
                if len(agr_names) != len(set(agr_names)):
                    raise ValueError(f"Duplicated aggregate levels for '{lvl}'")
                
                for agr_name in agr_names:
        
                    if agr_name is None:
                        qry_lvls_sel += [f"lvl_{lvl_id}.value AS {lvl}"]
                        qry_lvls_group += [lvl]
                    else:
                        res = cur.execute("""
                            SELECT rel_id FROM relation_tables
                                WHERE lvl_name = ? AND agr_name = ?
                        """, (lvl, agr_name))
        
                        try:
                            rel_id = res.fetchone()[0]
                        except TypeError:
                            raise ValueError(f"'{agr_name}' as an aggregated level for '{lvl}' not found")
        
                        qry_lvls_join += [f"LEFT JOIN rel_{rel_id} ON lvl_{lvl_id}.value = rel_{rel_id}.lvl_value"]
                        qry_lvls_sel += [f"rel_{rel_id}.agr_value AS {agr_name}"]
                        qry_lvls_group += [agr_name]
            
            qry_lvls_group = ", ".join(['scn', 'year'] + qry_lvls_group)
            qry_lvls_sel = """,
                    """.join(qry_lvls_sel) + ',' if qry_lvls_sel else ''
            qry_lvls_join = """
                """.join(qry_lvls_join)
        
            if isinstance(scn, str) and scn != 'all':
                scn = [scn]
            if isinstance(years, str|int) and years != 'all':
                years = [years]
        
            qry_where = ""
            if isinstance(scn, list) or isinstance(years, list):
                qry_where += "WHERE "
                if isinstance(scn, list):
                    qry_where += f"""scn IN ({', '.join(f'"{s}"' for s in scn)})"""
                    if isinstance(years, list):
                        qry_where += " AND "
                if isinstance(years, list):
                    qry_where += f"""year IN ({', '.join(f'"{y}"' for y in years)})"""
                
            qry = f"""
                SELECT 
                    s.name AS scn,
                    year,
                    {qry_lvls_sel}
                    SUM(d.value) AS value
                FROM
                    attr_{attr_id} AS d
                {qry_lvls_join}
                LEFT JOIN runs AS r ON d.run_id = r.run_id
                LEFT JOIN scenarios AS s ON r.scn_id = s.scn_id
                {qry_where}
                GROUP BY
                    {qry_lvls_group}
                ORDER BY
                    s.scn_id, year
            """

            if qry in self.cache:

                # Read from cache
                df = self.cache[qry]
                
            else:
                
                # print(qry)  
                try:
                    res = pd.read_sql_query(
                        con = con,
                        sql = qry
                    )
                except:
                    sel = con.execute(f"""SELECT * FROM attr_{attr_id}""")
                    lvls_in_table = [c[0] for c in sel.description if 'lvl_' in c[0]]
                    group_lvls_not_in_table = [n for n,id in zip(groupby,lvl_ids) if 'lvl_'+str(id) not in lvls_in_table]
                    if group_lvls_not_in_table:
                        raise ValueError(f"Group by level(s) not in data table: {group_lvls_not_in_table}")
                    else:
                        raise Exception("Something went wrong getting data from database file")
                    
                df = (
                    res
                    .set_index(list(res.columns)[0:-1])
                    ['value']
                )
                    
                if df.index.nlevels > 2:
                    df = (
                        df
                        .unstack(['scn','year'], fill_value=0)
                        .T
                    )

                if len(df) == 0:
                    # If no data returned add in expected index levels (scn, year)
                    qry = f"""
                        SELECT
                            s.name AS scn,
                            r.year AS year
                        FROM
                            runs AS r
                        LEFT JOIN scenarios AS s ON r.scn_id = s.scn_id
                        {qry_where + ("WHERE" if qry_where == "" else " AND")} r.calculated != 0
                        ORDER BY
                            s.scn_id, year
                    """
                    res = cur.execute(qry)
                    df.index = pd.MultiIndex.from_tuples(
                        res.fetchall(),
                        names = ['scn', 'year']
                    )

                # Add to cache
                self.cache[qry] = df
            
        if all_region_levels and 'region' in groupby:
            with closing(sqlite3.connect(self.db_path, timeout=self.db_timeout)) as con, con,  \
                closing(con.cursor()) as cur:
                
                # Get all region levels present in df
                res = cur.execute("""
                    SELECT agr_name FROM relation_tables
                        WHERE lvl_name = "region"
                """)
                    
                reg_lvls = ['region'] if 'region' in df.columns.names else []
                reg_lvls += [r[0] for r in res if r[0] in df.columns.names]
                other_lvls = list(set(df.columns.names) - set(reg_lvls))

                # Construct region index with all level values
                qry_sel = ["lvl.value AS region"]
                qry_join = ""

                res = cur.execute("""
                    SELECT lvl_id FROM levels
                        WHERE name = "region"
                """)
                reg_lvl_id = res.fetchone()[0]

                res = cur.execute("""
                    SELECT rel_id, agr_name FROM relation_tables
                        WHERE lvl_name = "region"
                """)

                for rel_id, agr_name in res.fetchall():
                    qry_sel += [f"rel_{rel_id}.agr_value AS {agr_name}"]
                    qry_join += f"LEFT JOIN rel_{rel_id} ON rel_{rel_id}.lvl_value = lvl.value"

                qry = f"""
                    SELECT
                        {", ".join(qry_sel)}
                    FROM
                        lvl_{reg_lvl_id} AS lvl
                    {qry_join}
                    """

                idx_df = pd.read_sql_query(
                    con = con,
                    sql = qry
                )

                reg_uni = idx_df.set_index(reg_lvls).index.unique()
                    
                if other_lvls:
                    other_uni = df.columns.droplevel(reg_lvls).unique()
                        
                    new_idx = pd.MultiIndex.from_tuples(
                        [(re if isinstance(re, tuple) else (re,)) +
                         (ot if isinstance(ot, tuple) else (ot,)) 
                         for re in reg_uni for ot in other_uni],
                        names = reg_uni.names + other_uni.names
                    ).reorder_levels(df.columns.names).sort_values()
                else:
                    new_idx = reg_uni
                    if isinstance(new_idx, pd.MultiIndex):
                        new_idx = new_idx.reorder_levels(df.columns.names).sort_values()
                        
                df = df.reindex(columns = new_idx, fill_value = 0)
        
        if interpolate:
            # Interpolate to yearly data
        
            # Create new index with all years represented
            new_idx = pd.MultiIndex.from_tuples(
                [
                    (scn,year)
                    for scn in df.index.unique('scn')
                    for year in range(
                        min(df.loc[scn].index),
                        max(df.loc[scn].index)+1
                    )
                ],
                names = ['scn','year']
            )
            # Reindex and interpolate
            df = df.reindex(new_idx).interpolate()
        
        # Change year level to string
        if len(df)>0:
            df.index = df.index.set_levels(df.index.levels[-1].astype(str), level=-1)
        
        # print('Retrieved data from', '.'.join([module, attr]))
        
        return df

class CacheDict(dict):
    # Implementation slightly modified from
    # https://gist.github.com/bencharb/729971d4a9e4633ea08a
    
    default_max_size = 100
    def __init__(self, *args, **kwargs):
        self.max_size = kwargs.pop('max_size', self.default_max_size)
        super(CacheDict, self).__init__(*args, **kwargs)
        
    def __setitem__(self, key, val):
        if self.max_size > 0:
            if key not in self:
                max_size = self.max_size-1  # so the dict is sized properly after adding a key
                self._prune_dict(max_size)
            super(CacheDict, self).__setitem__(key, val)
        
    def update(self, **kwargs):
        if self.max_size > 0:
            super(CacheDict, self).update(**kwargs)
            self._prune_dict(self.max_size)

    def _prune_dict(self, max_size):
        if len(self) >= max_size:
            diff = len(self) - max_size 
            for k in list(self)[:diff]:
                del self[k]

    def __reduce__(self):
        # Used in pickling/unpickling
        # Content of dict is dropped
        return (self.__class__, (), self.__dict__.copy())

def _isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def _level_names_to_integer_key(data, db_path, timeout):

    data = data.copy()
    
    # Get index and columns levels
    lvls_axs = [(lvl, 0) for lvl in data.index.names]
    if isinstance(data, pd.DataFrame):
        lvls_axs += [(lvl, 1) for lvl in data.columns.names]

    with closing(sqlite3.connect(db_path, timeout=timeout)) as con, con,  \
        closing(con.cursor()) as cur:

        lvl_keys = []
        for lvl, ax in lvls_axs:

            cur.execute("""
                INSERT OR IGNORE
                    INTO levels(name)
                    VALUES(?)
            """, (lvl,))

            res = cur.execute("""
                SELECT lvl_id FROM levels
                    WHERE name = ?
            """, (lvl,))

            lvl_id = res.fetchone()[0]
            lvl_keys += ['lvl_'+str(lvl_id)]
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS lvl_{lvl_id} (
                    lvl_value_id INTEGER PRIMARY KEY,
                    value TEXT UNIQUE
                );
            """)

            while True:
                res = cur.execute(f"""
                    SELECT value, lvl_value_id FROM lvl_{lvl_id}
                """)
                txt_to_int = dict(res.fetchall())

                data = data.rename(txt_to_int, axis=ax, level=lvl)

                if ax == 0:
                    if len(data.index) == 0:
                        lvl_dtype = int
                    else:
                        lvl_dtype = data.index.get_level_values(lvl).dtype
                else:
                    if len(data.columns) == 0:
                        lvl_dtype = int
                    else:
                        lvl_dtype = data.columns.get_level_values(lvl).dtype
                    
                if pd.api.types.is_integer_dtype(lvl_dtype):
                    break
                    
                if ax == 0:
                    unique_lvl_values = data.index.unique(lvl)
                else:
                    unique_lvl_values = data.columns.unique(lvl)
                
                to_write = [(v,) for v in unique_lvl_values]
                cur.executemany(f"""
                    INSERT OR IGNORE
                        INTO lvl_{lvl_id}(value)
                        VALUES(?)
                """, to_write)
                
        if isinstance(data, pd.DataFrame):
            # Stack to series (take shortest way)
            data = data.dropna(axis=1, how='all')
            if data.shape[1] == 0:
                # If no columns remain (i.e. all values in dataframe were NaN)
                # return an empty pd.Series with the correct levels
                return pd.Series(
                        index = pd.MultiIndex.from_tuples(
                            [],
                            names = lvl_keys
                        )
                    )
            if data.columns.nlevels > data.index.nlevels:
                data = (
                    data.unstack(data.index.names)
                    .reorder_levels(data.index.names+data.columns.names)
                    .sort_index()
                )
            else:
                if data.columns.nlevels == 1 and isinstance(data.columns, pd.MultiIndex):
                    # Fix problem with single-level MultiIndex stacking by
                    # converting to Index
                    data.columns = data.columns.get_level_values(0)
                data = data.stack(data.columns.names)
            
        data = data.rename_axis(index = lvl_keys)
        data = data.dropna()
        
        assert isinstance(data, pd.Series)
        assert not data.isna().any()

        return data
    

def _get_check_and_clean_data(module, module_name, attr, zero_tol=1e-6):

    data = module.data_attr.get(attr).copy()
    
    if isinstance(data, pd.Series|pd.DataFrame):
            
        if data.isna().any().any():
            warnings.warn(f'NaNs in {module.par.name}.{attr}.')
        if (data < -zero_tol).any().any():
            warnings.warn(f'Negative values of down to {data.min().min()} {module.data_attr[attr]["unit"]} in {module_name}.{attr}.')
        
        # Set zeros to NaN
        data = data.where(data >= zero_tol, np.nan)
        
    elif isinstance(data, np.float_):
        
        if np.isnan(data):
            warnings.warn(f'NaNs in {module.par.name}.{attr}.')
            data = 0
        if data < -zero_tol:
            warnings.warn(f'Negative value of {data} {module.data_attr[attr]["unit"]} in {module_name}.{attr}.')
        if data < zero_tol:
            data = 0
    
    else:
        raise TypeError(f"Data attribute '{attr}' not a pandas.Series, pandas.DataFrame or numpy.float")

    return data

def _parse_datetime_str(str):
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str, fmt)
        except ValueError:
            try:
                today_date = date.today()
                return datetime.strptime(f"{today_date} {str}", fmt)
            except ValueError:
                pass
    raise ValueError('Unrecognized datetime format')