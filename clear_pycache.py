#%% Remove caches:
import pathlib

[p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]
[p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]
[p.rmdir() for p in pathlib.Path('.').rglob('__pycache__ (1)')]
[p.rmdir() for p in pathlib.Path('.').rglob('__pycache__ (2)')]
 
#%%
