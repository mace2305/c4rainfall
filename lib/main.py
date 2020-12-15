"""
main user interface for cli run
"""
import utils, modellers
import sys
import numpy as np
import logging


# from utils import parse_args

domains_inter_mon = [
    #[-10.0, 20.0, 90.0, 150.0], [-12.0, 28.0, 90.0, 140.0], [-13.5, 29.5, 92.0, 138.0], [-4.0, 8.0, 93.5, 114.0], 
    #[-5.0, 9.0, 95.0, 112.5], [-6.0, 10.0, 82.5, 127.5]    
]

domains_SW_mon = [
    #[-4.0, 8.0, 93.5, 114.0], [-5.0, 9.0, 95.0, 112.5], [-6.0, 10.0, 82.5, 127.5], [-6.0, 10.0, 96.0, 111.5], 
    #[-10.0, 26.0, 87.0, 143.0], [-12.0, 16.0, 92.0, 118.0], [-12.0, 28.0, 90.0, 140.0], [-13.5, 29.5, 92.0, 138.0], 
    #[-20.0, 20.0, 75.0, 125.0], 
    [-20.0, 20.0, 75.0, 150.0], [-20.0, 20.0, 90.0, 140.0], [-20.0, 25.0, 75.0, 142.0], 
    [-20.0, 30.0, 90.0, 130.0], [-20.0, 35.0, 75.0, 130.0], [-30.0, 25.0, 65.0, 147.0], [-30.0, 25.0, 75.0, 130.0], 
    [-30.0, 30.0, 65.0, 140.0], [-30.0, 32.0, 75.0, 148.0], [-30.0, 42.0, 75.0, 138.0]
]

domains_NE_mon = [
    #[-10.0, 20.0, 90.0, 150.0], [-10.0, 26.0, 87.0, 143.0], [-13.5, 29.5, 92.0, 138.0], [-16.0, 32.0, 94.0, 136.0], 
    #[-30.0, 18.0, 75.0, 168.0], [-30.0, 32.0, 75.0, 148.0], [-30.0, 42.0, 75.0, 138.0], [-5.0, 9.0, 95.0, 112.5], 
    #[-6.0, 10.0, 96.0, 111.5], 
    #[-10.0, 14.0, 90.0, 120.0]
]

hpparam = [60000, 16, 'train_batch', 4, .15, 0]

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(f'{utils.logs_dir}/runtime_{utils.datetime_now()}.log', 'a'))
print = logger.info

#seq_strings = "NE_mon", "SW_mon", "inter_mon"
#for i,d in enumerate((domains_NE_mon, domains_SW_mon, domains_inter_mon)):
    #perms = [(dims, seq_strings[i], hpparam) for dims in d]
    #print(f'\nFor {seq_strings[i]}:')
    #for p in perms: print(f'{p}, ')
    #for p in perms:
        #try:
            #model = modellers.TopLevelModel(p[0], p[1], p[2])
            #model.train_SOM()
            #model.detect_som_products()
            #model.generate_k()
        #except:
            #logger.info('%(message)s', exc_info=True)
            #continue
        
seq_strings = "SW_mon"
perms = [(dims, seq_strings, hpparam) for dims in domains_SW_mon]
print(f'\nFor {seq_strings}:')
for p in perms: print(f'{p}, ')
for p in perms:
    try:
        model = modellers.TopLevelModel(p[0], p[1], p[2])
        model.train_SOM()
        model.detect_som_products()
        model.generate_k()
    except:
        logger.info('%(message)s', exc_info=True)
        continue
