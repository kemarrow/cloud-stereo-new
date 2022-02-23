import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from mask2 import mask_C1, mask_C3


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage

#to install: pip install ptlflow

# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset

date = '2021-09-28_15A'
vidcapR = cv.VideoCapture('Videos/lowres_C1_'+ date+'.mp4')
vidcapL = cv.VideoCapture('Videos/C3_'+ date+'.mp4')
cap = cv.VideoCapture('Videos/C3_'+date+'.mp4')

model = ptlflow.get_model('flownet2', pretrained_ckpt='things') #other models are here: https://ptlflow.readthedocs.io/en/latest/models/models_list.html

#cap2 = cv.VideoCapture('depth_10_24_12A.mp4')
ret, frame1 = cap.read()
#ret2, img = cap2.read()

success, imgR = vidcapR.read()
success2, imgL = vidcapL.read()

prvsR1 = imgR
prvsL1 = imgL

#prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    success, imgR = vidcapR.read()
    success2, imgL = vidcapL.read() 
    # Load the images
    images = [prvsL1, imgL]

    images[0] = prvsL1
    images[0] = imgL

    # A helper to manage inputs and outputs of the model
    io_adapter = IOAdapter(model, prvsL1.shape[:2])

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs([prvsL1, imgL])

    # Forward the inputs through the model
    predictions = model(inputs)

    # Remove extra padding that may have been added to the inputs
    predictions = io_adapter.unpad_and_unscale(predictions)

    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']
    flowsnet  =  flows.detach().numpy()
   
    #flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    #print(flows.shape)
    #print(flows[3])
    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    print(np.shape(flow_rgb))
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()


    flownet  = cv.cvtColor(flow_rgb_npy, cv.COLOR_BGR2GRAY)
    print(np.shape(flow_rgb_npy))

    # OpenCV uses BGR format
    maskcomb =   mask_C3[:,:,0]
    rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
    new_mask =  maskcomb & (rg_ratio<1.1) & (rg_ratio>0.93)#maskL & thresh1
    
    #rg_ratio1 = prvsL1[:, :, 1]/prvsL1[:, :, 2]
    #new_mask1 =  maskcomb & (rg_ratio1<1.1) & (rg_ratio1>0.93)

    flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
    #flow_bgr_npym = cv.bitwise_and(flow_bgr_npy, flow_bgr_npy, mask=new_mask)
    #______________
    img = cv.cvtColor(images[1], cv.COLOR_BGR2GRAY)
    img1 = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
    #next = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(img1, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    thresh1 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    ret,thresh1 = cv.threshold(thresh1,1,255,cv.THRESH_BINARY)
    #newthing =  cv.bitwise_and(rimgL, rimgL, mask=thresh1)
    #thresh1[thresh1==255] = None


    flowsnet= flowsnet.squeeze()
    flowsnet  = np.swapaxes(flowsnet, 0, 1)
    flowsnet  = np.swapaxes(flowsnet, 1, 2)

    differencex =  flowsnet[:,:,0] -  flow[:,:,0]
    differencey =  flowsnet[:,:,1] -  flow[:,:,1]

    differencexm = cv.bitwise_and(differencex, differencex, mask=thresh1) #new_mask
    differencexm[differencexm == thresh1] = None

    differenceym = cv.bitwise_and(differencey, differencey, mask=thresh1) #new_mask
    differenceym[differenceym == thresh1] = None

    speed1Dflownet = np.sqrt(flowsnet[:,:,0]**2 + flowsnet[:,:,1]**2)
    speed1Dflow = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    difference_speed = speed1Dflownet - speed1Dflow

    difference_speed_masked = cv.bitwise_and(difference_speed, difference_speed, mask=thresh1) #new_mask
    difference_speed_masked[difference_speed_masked == thresh1] = None #new_mask

    #difference_speed_masked = cv.bitwise_and(rimgL, rimgL, mask=thresh1)
    # Show on the screen
    #dif  = flow_bgr_npy - bgr
    #cv.imshow('difference', dif)
    #cv.imshow('image2', images[1])
    #cv.imshow('flow', flow_bgr_npy)

    fig2, ((ax ,ax1), (ax2, ax3)) = plt.subplots(2,2, sharex = True, sharey = True)
    limitx = max(abs(differencex.flatten()))
    limity = max(abs(differencey.flatten()))
    disp = ax.imshow(differencex,'coolwarm', vmin = -limitx, vmax = limitx)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(disp, cax=cax, orientation='vertical')
    ax.set_xlabel('difference in horizontal (pixel)')

    disp = ax2.imshow(differencexm,'coolwarm', vmin = -limitx, vmax = limitx)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(disp, cax=cax, orientation='vertical')
    ax2.set_xlabel('difference in horizontal (pixel)')


    disp = ax1.imshow(differencey,'coolwarm', vmin = -limity, vmax = limity)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(disp, cax=cax, orientation='vertical')

    ax1.set_xlabel('difference in vertical (pixel)')

    disp = ax3.imshow(differenceym,'coolwarm', vmin = -limity, vmax = limity)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(disp, cax=cax, orientation='vertical')

    ax3.set_xlabel('difference in vertical (pixel)')
    plt.show()

      
    #difference_speed[speed1Dflownet  == 0 ] = None
    
    fig, (ax ,ax1,ax2, ax3, ax4) = plt.subplots(1,5, sharex = True, sharey = True)
    images[0] = cv.cvtColor(images[0], cv.COLOR_BGR2RGB)
    ax.imshow(images[0])
    ax.set_xlabel('Original')
    ax3.set_xlabel('Difference in magnitude (pixel)')
    ax1.imshow(bgr)
    ax1.set_xlabel('Farneback algorithm')
    ax2.imshow(flow_bgr_npy)
    ax2.set_xlabel('Flownet')

    limitspeed = max(abs(difference_speed.flatten()))

    disp = ax3.imshow(difference_speed,'coolwarm', vmin= -limitspeed, vmax = limitspeed)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(disp, cax=cax, orientation='vertical')


    disp4 = ax4.imshow(difference_speed_masked,'coolwarm', vmin= -limitspeed, vmax = limitspeed)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(disp4, cax=cax4, orientation='vertical')
    ax4.set_xlabel('Difference with mask')
    plt.show()

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
   
    prvs = next
    prvsR1 = imgR
    prvsL1 = imgL
cv.destroyAllWindows()