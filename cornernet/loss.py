import torch
import torch.nn as nn

residual = torch.tensor(1e-4)


class Loss(nn.Module):

    def __init__(self, pull_weight=0.1, push_weight=0.1, off_weight=1):
        super(Loss, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight

        self.focal_loss  = refined_focal_loss
        self.offset_loss = offset_loss
        self.pull_push_loss = pull_push_loss

    def forward(self, pred, gt):
        tl_heats   = pred[0]
        tl_embeds  = pred[1]
        tl_offsets = pred[2]
        br_heats   = pred[3]
        br_embeds  = pred[4]
        br_offsets = pred[5]

        gt_tl_heats   = gt[0]
        gt_br_heats   = gt[1]
        gt_tl_offsets = gt[2]
        gt_br_offsets = gt[3]
        gt_tl_pos     = gt[4]
        gt_br_pos     = gt[5]
        gt_obj_mask   = gt[6]

        # refined focal loss
        focal_loss = 0
        tl_heats = torch.clamp(torch.sigmoid(tl_heats), min=1e-4, max=1-1e-4)
        br_heats = torch.clamp(torch.sigmoid(br_heats), min=1e-4, max=1-1e-4)

        focal_loss += self.focal_loss(tl_heats, gt_tl_heats)
        focal_loss += self.focal_loss(br_heats, gt_br_heats)

        # offsets loss
        off_loss = 0

        off_loss += self.offset_loss(tl_offsets, gt_tl_offsets, gt_tl_pos, gt_obj_mask)
        off_loss += self.offset_loss(br_offsets, gt_br_offsets, gt_br_pos, gt_obj_mask)

        off_loss = self.off_weight * off_loss

        # pull and push loss
        pull_loss, push_loss = self.pull_push_loss(tl_embeds, br_embeds, gt_tl_pos, gt_br_pos, gt_obj_mask)
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        #print("focal loss: {}  offsets loss: {}  pull loss: {}  push loss: {}".format(focal_loss, off_loss, pull_loss, push_loss))

        loss = (focal_loss + off_loss + pull_loss + push_loss) / len(tl_heats)
        #loss = (focal_loss + off_loss) / len(tl_heats)

        return loss.unsqueeze(0)


def refined_focal_loss(pred, gt, alpha=2, beta=4):
    loss = 0
    batch = gt.size()[0]

    for i in range(batch):
        pos_inds = gt[i].eq(1)
        neg_inds = gt[i].lt(1)

        pos_pred = pred[i][pos_inds]
        pos_loss = torch.pow(1-pos_pred, alpha) * torch.log(pos_pred)

        neg_weight = torch.pow(1 - gt[i][neg_inds], beta)
        neg_pred = pred[i][neg_inds]
        neg_loss = neg_weight * torch.pow(neg_pred, alpha) * torch.log(1 - neg_pred)

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        num_pos = pos_inds.float().sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def offset_loss(pred, gt, pos, mask):
    loss = 0
    batch = gt.size()[0]
    pos = pos.to(bool)
    for i in range(batch):
        obj_num = mask[i]
        if obj_num != 0:
            offsets = pred[i]   # 2,128,128
            gt_offsets = gt[i]  # 2,128,128
            pos_inds = pos[i]

            x = offsets[0][pos_inds]
            y = offsets[1][pos_inds]

            gt_x = gt_offsets[0][pos_inds]
            gt_y = gt_offsets[1][pos_inds]

            x_loss = nn.functional.smooth_l1_loss(x, gt_x, size_average=False)
            y_loss = nn.functional.smooth_l1_loss(y, gt_y, size_average=False)

            #loss += (x_loss + y_loss) / (obj_num + 1e-4)
            loss += (x_loss + y_loss) / (obj_num + residual)
        else:
            loss = 0
    return loss


def pull_push_loss(tl, br, tl_pos, br_pos, mask):
    pull = 0
    push = 0
    batch = tl.size()[0]
    #tl_pos = tl_pos.to(bool)
    #br_pos = br_pos.to(bool)
    for i in range(batch):
        obj_num = mask[i]

        if obj_num != 0:
            tl_ = tl[i].squeeze()
            br_ = br[i].squeeze()
            tl_inds = tl_pos[i]
            br_inds = br_pos[i]

            tl_list = torch.tensor([0], dtype=torch.float32, device='cuda')
            br_list = torch.tensor([0], dtype=torch.float32, device='cuda')
            for num in range(obj_num):
                tl_ind = tl_inds.eq(num+1)
                br_ind = br_inds.eq(num+1)
                if len(tl_[tl_ind])==0 or len(br_[br_ind])==0:
                    pass
                else:
                    tl_list = torch.cat((tl_list, tl_[tl_ind]))
                    br_list = torch.cat((br_list, br_[br_ind]))

            mean = (tl_list + br_list) / 2

            pull_tl = torch.pow(tl_list - mean, 2).sum()
            pull_br = torch.pow(br_list - mean, 2).sum()
            #pull += (pull_tl + pull_br) / (obj_num + 1e-4)
            pull += (pull_tl + pull_br) / (obj_num + residual)

            mean = mean[1:]
            dist = 0
            for num in range(len(mean)):
                ek = mean[num]
                x = nn.functional.relu(1 - torch.abs(ek - mean), inplace=True)
                dist += x.sum() - 1

            obj_nums = obj_num*(obj_num-1)
            #push += dist / (obj_nums + 1e-4)
            push += dist / (obj_nums + residual)
        else:
            push = 0
            pull = 0

    return pull, push
'''
def offset_loss(pred, gt, pos, mask):
    loss = 0
    batch = gt.size()[0]
    pos = pos.to(bool)
    for i in range(batch):
        offsets = pred[i]
        gt_offsets = gt[i]

        obj_num = mask[i]

        img_loss = 0
        for j in range(obj_num):
            x = pos[i, j, 0]
            y = pos[i, j, 1]

            img_loss += nn.functional.smooth_l1_loss(offsets[:, y, x], gt_offsets[:, y, x])

        img_loss = img_loss / obj_num
        loss += img_loss

    return loss



def pull_push_loss(tl, br, tl_pos, br_pos, mask):
    pull = 0
    push = 0
    batch = tl.size()[0]

    for i in range(batch):
        obj_num = mask[i]

        img_loss = 0
        for j in range(obj_num):
            tlx = tl_pos[i, j, 0]
            tly = tl_pos[i, j, 1]
            brx = br_pos[i, j, 0]
            bry = br_pos[i, j, 1]

            tl_embeds = tl[i, :, tly, tlx]
            br_embeds = br[i, :, bry, brx]

            mean = (tl_embeds + br_embeds) / 2

            img_loss += torch.pow(tl_embeds-mean, 2) + torch.pow(br_embeds-mean, 2)

        img_loss = img_loss / obj_num
        pull += img_loss

    return pull, push
'''
'''
def offset_loss(pred, gt, pos, mask):
    loss = 0
    batch = gt.size()[0]
    for i in range(batch):
        offsets = pred[i]
        gt_offsets = gt[i]

        obj_num = mask[i]

        x = pos[i, :obj_num, 0]
        y = pos[i, :obj_num, 1]

        offsets = torch.index_select(offsets, 1, y)
        offsets = torch.index_select(offsets, 2, x)

        gt_offsets = torch.index_select(gt_offsets, 1, y)
        gt_offsets = torch.index_select(gt_offsets, 2, x)

        loss += nn.functional.smooth_l1_loss(offsets, gt_offsets, size_average=False) / obj_num

    return loss


def pull_push_loss(tl, br, tl_pos, br_pos, mask):
    pull = 0
    push = 0

    batch = tl.size()[0]

    for i in range(batch):
        obj_num = mask[i]
        tl_embeds = tl[i]
        br_embeds = br[i]

        tlx = tl_pos[i, :obj_num, 0]
        tly = tl_pos[i, :obj_num, 1]
        tl_embeds = torch.index_select(tl_embeds, 1, tly)
        tl_embeds = torch.index_select(tl_embeds, 2, tlx)

        brx = br_pos[i, :obj_num, 0]
        bry = br_pos[i, :obj_num, 1]
        br_embeds = torch.index_select(br_embeds, 1, bry)
        br_embeds = torch.index_select(br_embeds, 2, brx)

        mean = (tl_embeds + br_embeds) / 2

        pull += torch.pow(tl_embeds - mean, 2) + torch.pow(br_embeds - mean, 2)


    return pull,push
'''