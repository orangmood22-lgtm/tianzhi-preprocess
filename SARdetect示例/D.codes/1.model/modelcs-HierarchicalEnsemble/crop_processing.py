import cv2, numpy as np, math
def pixels(arr,box,quad,mode):
    if mode=='aligned':
        (cx,cy),(w,h),angle=cv2.minAreaRect(np.asarray(quad,np.float32))
        if h>w:w,h,angle=h,w,angle+90
        width=max(8,int(math.ceil(w*1.25)));height=max(8,int(math.ceil(h*1.25)))
        matrix=cv2.getRotationMatrix2D((cx,cy),angle,1)
        matrix[0,2]+=width/2-cx;matrix[1,2]+=height/2-cy
        crop=cv2.warpAffine(arr,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    else:
        x,y,w,h=box;cx=x+w/2;cy=y+h/2;w=max(w*1.25,8);h=max(h*1.25,8)
        x0=max(0,int(math.floor(cx-w/2)));y0=max(0,int(math.floor(cy-h/2)))
        x1=min(arr.shape[1],int(math.ceil(cx+w/2)));y1=min(arr.shape[0],int(math.ceil(cy+h/2)))
        if x1<=x0 or y1<=y0:raise ValueError('Empty crop')
        crop=arr[y0:y1,x0:x1]
    lo,hi=np.percentile(crop,[1,99]);crop=np.clip((crop.astype(np.float32)-lo)/max(hi-lo,1),0,1)
    if mode=='hbb':return cv2.resize(crop,(160,160))
    h,w=crop.shape;s=160/max(h,w);hh=max(1,round(h*s));ww=max(1,round(w*s))
    fitted=cv2.resize(crop,(ww,hh));result=np.zeros((160,160),np.float32)
    y=(160-hh)//2;x=(160-ww)//2;result[y:y+hh,x:x+ww]=fitted
    return result
