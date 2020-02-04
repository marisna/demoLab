call  mkvirtualenv envOpenpose & workon envOpenpose
call pip install -r requirements.txt 
call pip install -r ..\openpose\requirements.txt 
python realtimeDetect.py --video=1 --thr=0.8
