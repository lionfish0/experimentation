import cv2
import yolov5

def getlocalpred(frame):
    results = model(frame)
    predictions = results.pred[0]
    ret = []
    for pred in predictions:
        box = pred[:4].cpu().numpy().astype(float)
        score = pred[4].cpu().numpy().item()
        cat = int(pred[5].cpu().numpy().item())
        item = {'box':box,'score':score,'cat':cat}
         # x1, y1, x2, y2
        ret.append(item)
    return ret

import numpy as np

import matplotlib.pyplot as plt

def getresultslikebox(preds,x1,y1,x2,y2):
    return [p for p in preds if np.all((p['box'][:2]+30>[x1,y1]) & (p['box'][2:]-30<[x2,y2]))] 

def getbestattack(frame,x1,y1,x2,y2,boxspace = 0):
    besttagspot = None
    minv = 1
    #adframe = frame[y1-boxspace:y2+boxspace,x1-boxspace:x2+boxspace].copy()
    adframe = frame.copy()
    preds = getlocalpred(adframe)
    preds = getresultslikebox(preds,x1,y1,x2,y2)
    if len(preds)==0:
        return None, None
    originalcla = preds[0]['cat']
    #for i in np.arange(boxspace,x2-x1,30):
    for i in np.arange(x1,x2,40):
        #for j in np.arange(boxspace,y2-y1,30):
        for j in np.arange(y1,y2,40):
            #adframe = frame[y1-boxspace:y2+boxspace,x1-boxspace:x2+boxspace].copy()
            adframe = frame.copy()
            adframe[j:j+10,i:i+10,:]=50 #a blackish sticker
            newpred = getlocalpred(adframe)
            newpred = getresultslikebox(newpred,x1,y1,x2,y2)
            #print(i,j,newpred[0]['score'])
            if len(newpred)>0:
                conf = newpred[0]['score']
                cla = newpred[0]['cat']
                if cla!=preds[0]['cat']:
                    minv=0
                    besttagspot = np.array([i,j])#+np.array([x1,y1])-boxspace
                    return None, besttagspot
                if conf<minv:
                    minv = conf
                    besttagspot = np.array([i,j])#+np.array([x1,y1])-boxspace
            
    #print(minv,besttagspot)
    return minv,besttagspot   


catstrs = ['no motor','nopriority','speed40']
#load custom model
model = yolov5.load('best.pt')
  
# set model parameters
model.conf = 0.15  # NMS confidence threshold
model.iou = 0.35  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

def drawbox(img,x1,y1,x2,y2,st):
    #print(x1,y1,x2,y2)
    cv2.putText(drawnframe,st, (x1,y1),  cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255), 1,2)    
    #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),1,2,0)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),1)
ct = 0
storedbesttagspots = [None,None,None]
while rval:
    
    drawnframe = frame.copy()
    
    if ct%20==0:
        ret = getlocalpred(frame)
    
    for r in ret:
        if r['score']<0.25: continue
        
        x1,y1,x2,y2 = r['box'].astype(int)
        drawbox(drawnframe,x1,y1,x2,y2,"%s (%0.2f)" % (catstrs[r['cat']],r['score']))
    
    
    for r in ret: 
        if r['score']<0.25: continue
        x1,y1,x2,y2 = r['box'].astype(int)
        try:
            if ct%200==0:
                minv, besttagspot = getbestattack(frame,x1,y1,x2,y2)
                if besttagspot is not None:
                    storedbesttagspots[r['cat']] = besttagspot
                #print(minv,besttagspot)
        except ZeroDivisionError:
            print("ZeroDivisionError :/")
    ct+=1  
    
    for bts in storedbesttagspots:
        #print(bts)
        if bts is not None:
            cv2.rectangle(drawnframe,(bts[0],bts[1]),(bts[0]+10,bts[1]+10),(0,0,255),1)
    cv2.imshow("preview", drawnframe)
    rval, frame = vc.read()
        
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
