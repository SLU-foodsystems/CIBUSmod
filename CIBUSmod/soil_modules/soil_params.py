C_CONTENT_CROPS = 0.5
SOM_to_SOC = 1/1.72

# GWP  values
GWP = {'gwp20': {'ipcc1995': {'ch4': 56,
                               'n2o': 280,
                               'co2': 1},
                  'ipcc2007': {'ch4': 72,
                               'n2o': 289,
                               'co2': 1},
                  'ipcc2013': {'ch4': 84,
                               'ch4-bio': 85,
                               'n2o': 264,
                               'co2': 1}
                 },
       'gwp100': {'ipcc1995': {'ch4': 21,
                               'n2o': 310,
                               'co2': 1},
                  'ipcc2007': {'ch4': 25,
                               'n2o': 298,
                               'co2': 1},
                  'ipcc2013': {'ch4': 28,
                               'ch4-bio': 30,
                               'n2o': 265,
                               'co2': 1}
                 },
       'gwp500': {'ipcc1995': {'ch4': 6.5,
                               'n2o': 170,
                               'co2': 1},
                  'ipcc2007': {'ch4': 7.6,
                               'n2o': 153,
                               'co2': 1},
                  'ipcc2013': {'ch4': float('nan'),
                               'n2o': float('nan'),
                               'co2': float('nan')} # GWP500 values not published in IPCCAR5
                 }
      }