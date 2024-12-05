from mp_api.client import MPRester
from emmet.core.xas import Edge, XASDoc, Type
import json

#Grabbing the data from the materials project 
with MPRester("od4YR72wa5gNURoZWLf8w1AFriPEIagz", monty_decode=False ,use_document_model=False) as mpr:
    xas_monty = mpr.materials.xas.search(fields=['structure', 'spectrum','formula_pretty'], elements=['V'], absorbing_element='V',edge = 'K')


# Extract only spectrum, structure, and formula
xas = {'structure': [k['spectrum']['structure'] for k in xas_monty],
       'spectrum': [k['spectrum'] for k in xas_monty],
       'formula_pretty': [k['formula_pretty'] for k in xas_monty]
      }


with open('data/V_XANES.json', 'w') as file:
    json.dump(xas, file)
