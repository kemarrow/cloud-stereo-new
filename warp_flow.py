import ptlflow

import cv2 as cv
import torch
import torch.nn.functional as F
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils
import numpy as np 
from matplotlib import pyplot as plt

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.tensor(grid) + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)

    return output



# Get an initialized model from PTLFlow
model = ptlflow.get_model('flownet2', 'things')
#model.eval()

# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
date = '2021-10-24_12A'
cap = cv.VideoCapture('Videos/C3_'+date+'.mp4')
success2, img1 = cap.read()
#ret, frame1 = cap.read()

#prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

while(1):
    # Load the two images
    ret, img2 = cap.read()

    #print(img1.shape())
    #print(img2.shape())

    io_adapter = IOAdapter(model, img1.shape[:2])
    inputs = io_adapter.prepare_inputs([img1, img2])

    


    # Forward the inputs to obtain the model predictions
    predictions = model(inputs)

    # Some padding may have been added during prepare_inputs. The line below ensures that the padding is removed
    # to make the predictions have the same size as the original images.
    predictions = io_adapter.unpad_and_unscale(predictions)
    
    #img2 = cv2.imread('Videos/tl4_2021-10-24_12A/out_im0002.png')
    # Visualize the predicted flow
    flow = predictions['flows'][0, 0]  #Remove batch and sequence dimensions

    img2_pt = torch.from_numpy(img2).float()
    out = warp(img2_pt.unsqueeze(0).permute(0,3,1,2), flow.unsqueeze(0).float())
    out = out.permute(0,2,3,1)[0,...].detach().cpu().numpy()

    fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex =True, sharey=True)
    

    img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
    out1= cv.cvtColor(out/255, cv.COLOR_RGB2BGR)
    out= cv.cvtColor(out, cv.COLOR_RGB2BGR)

    residual = np.abs(img2/255.0 - img1/255.0)
    residualwarp = np.abs(out/255.0 - img1/255.0)

    
    #residual = cv.cvtColor(residual, cv.COLOR_RGB2BGR)
    #residualwarp = cv.cvtColor(residualwarp, cv.COLOR_RGB2BGR)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(residual)
    ax4.imshow(img1)
    ax5.imshow(out1)
    ax6.imshow(residualwarp)
    ax1.set_yticklabels([])
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])

    ax1.set_yticks([])
    ax1.set_xticks([])

    ax2.set_yticks([])
    ax2.set_xticks([])

    ax3.set_yticks([])
    ax3.set_xticks([])

    ax4.set_yticks([])
    ax4.set_xticks([])

    ax5.set_yticks([])
    ax5.set_xticks([])

    ax6.set_yticks([])
    ax6.set_xticks([])
   
    ax1.set_xlabel('Frame 1')
    ax2.set_xlabel('Frame 2')
    ax3.set_xlabel('Residual')
    ax4.set_xlabel('Frame 1')
    ax5.set_xlabel('Frame 1 warped')
    ax6.set_xlabel('Residual of warped images')
   
    fig.tight_layout()
    plt.show()
    

    fig.savefig('Warping.eps', format='eps')

    #cv.imwrite('raw_residual.png',np.abs(img2/255.0 - img1/255.0)*255.0*2.0)
    #cv.imwrite('aligned_residual.png',np.abs(out/255.0 - img1/255.0)*255.0*2.0)
    #cv.imwrite('warped.png', out)
    #cv.imwrite('img1.png', img1)
    #cv.imwrite('img2.png', img2)

    img1= img2

# cv2.waitKey(0)

flow = flow.permute(1, 2, 0)  # change from CHW to HWC shape
flow = flow.detach().numpy()
flow_viz = flow_utils.flow_to_rgb(flow)  # Represent the flow as RGB colors
flow_viz = cv.cvtColor(flow_viz, cv.COLOR_BGR2RGB)  # OpenCV uses BGR format
cv.imwrite('flow.png',flow_viz)
# cv2.waitKey(0)
