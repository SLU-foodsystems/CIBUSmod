class DataAttr(object):
    '''Class that assigns data attributes on main modules and keeps track of metadata.
    
    Parameters
    ----------
    parent : A CIBUSmod main module
    '''

    def __init__(self, parent):
        self.parent = parent
        self.data = dict()
        self.metadata = dict()

    def __repr__(self):
        
        # Get column widths for printing
        tot_w = 140
        if len(self.metadata) > 0:
            name_w = max([len(n) for n in self.metadata]) + 3
            unit_w = max([len(self.metadata[n]['unit']) for n in self.metadata]) + 3
            orig_w = max([len(self.metadata[n]['orig']) for n in self.metadata]) + 3
        else:
            name_w = 30
            unit_w = 20
            orig_w = 30
        desc_w = tot_w - name_w - unit_w - orig_w

        fmt_str = "{:<"+str(name_w)+"} {:<"+str(unit_w)+"} {:<"+str(orig_w)+"} {:<"+str(desc_w)+"}\n"

        rep_str = fmt_str.format('ATTR', 'UNIT', 'ORIG', 'DESC')

        for key, value in self.metadata.items():
            name = key
            unit, orig, desc = value['unit'], value['orig'], value['desc']
            
            for i in range(max(
                round(len(name)/(name_w-2)+0.5),
                round(len(unit)/(unit_w-2)+0.5),
                round(len(orig)/(orig_w-2)+0.5),
                round(len(desc)/(desc_w-2)+0.5)
            )):
                rep_str += fmt_str.format(
                    name[(name_w-2)*i:(name_w-2)*(i+1)],
                    unit[(unit_w-2)*i:(unit_w-2)*(i+1)],
                    orig[(orig_w-2)*i:(orig_w-2)*(i+1)],
                    desc[(desc_w-2)*i:(desc_w-2)*(i+1)]
                )
                
        return rep_str

    def __getitem__(self, item):
        return self.metadata[item]

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        return iter(self.metadata)

    def keys(self):
        return self.metadata.keys()

    def items(self):
        return self.metadata.items()

    def values(self):
        return self.metadata.values()

    def add(
        self,
        data,
        name:str,
        unit:str = '',
        orig:str = '',
        desc:str = '',
        scalable:bool = True,
        allow_neg:bool = False
    ):
        '''Sets data attribute on main module (parent) and stores meta-data.

        Parameters
        ----------
        data : Any (usually pandas.DataFrame or pandas.Series)
            Data to store in data attribute
        name : str
            Name of data attribute
        unit : str, default ''
            Unit
        orig : str, default ''
            Origin of the data. A CIBUSmod module name where it is calculated
        desc : str, default ''
            Short description of data
        scalable : Bool, default True
            If scalable is True data is scaled in .scale() methods otherwise not
        allow_neg : Bool, default True
            If allow_neg is True warnings are not issued for negative values

        Returns
        -------
        None
        '''
        # Set attribute in parent
        self.data.update({name : data})
        # Update dict
        self.metadata.update({
            name : {
                'unit' : unit,
                'orig' : orig,
                'desc' : desc,
                'scalable' : scalable,
                'allow_neg' : allow_neg
            }
        })

        return None
    
    def update(self, name:str, data):
        '''Updates data attribute withot changing meta-data
        
        Parameters
        ----------
        name : str
            Name of data attribute
        data : Any (usually pandas.DataFrame or pandas.Series)
            Data to store in data attribute

        Returns
        -------
        None
        '''
        if name in self.metadata:
            self.data.update({name : data})
    
    def remove(self, attr:str):
        '''Remove data attribute
        
        Parameters
        ----------
        attr : str
            Data attribute to remove
            
        Returns
        -------
        None
        '''

        if attr in self.metadata:
            # Remove attribute from parent
            self.data.pop(attr)
            # Remove attribute from metadata
            self.metadata.pop(attr)
            return None
        else:
            raise KeyError(f'"{attr}"')

    
    def get(self, attr:str):
        '''Get data attribute
        
        Parameters
        ----------
        attr : str
            Data attribute to get
            
        Returns
        -------
        Data attribute, usually a pandas.DataFrame or pandas.Series'''

        return self.data[attr]