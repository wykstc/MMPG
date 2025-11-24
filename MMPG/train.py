import os
import numpy as np
import time
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import torch
from torch import nn
from model import MMpgNet
from utils import set_seed
from dataset import ECdataset
from dataset import FOLDdataset
from dataset import GOdataset
from dataset import RCdataset
from torch_geometric.data import DataLoader
import warnings
from utils import fmax
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

WEIGHTED_WEIGHT = 0.1

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Save the current model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc
    }
    save_path = os.path.join(save_dir, f'checkpoint.pt')
    torch.save(checkpoint, save_path)
    print(f"epoch {epoch}, best_val_acc={best_val_acc}, checkpoint saved to {save_path}")


def train(args, model, loader, optimizer, device):
    """
    train model for FOLD and FUNC task
    """
    model.train()
    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
        batch = batch.to(device)
        try:
            pred, o1, moe_loss = model(batch, True)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print('\n forward error \n')
                raise (e)
            else:
                print('OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))
        function = batch.y
        functions.append(function)
        optimizer.zero_grad()
        losso1 = F.nll_loss(o1.log_softmax(dim=-1), function)  # [B]
        loss = F.nll_loss(pred.log_softmax(dim=-1), function) + WEIGHTED_WEIGHT * losso1 + moe_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        loss_accum += loss.item()
    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds == functions) / functions.shape[0]
    return loss_accum / (step + 1), acc.item()

def evaluation(args, model, loader, device):
    """
    evaluation model for FOLD and FUNC task
    """
    model.eval()
    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        try:
            with torch.no_grad():
                pred, o1, moeloss = model(batch, False)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('evaluation OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))
        function = batch.y
        functions.append(function)
        loss = criterion(pred, function)
        loss_accum += loss.item()
    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    return loss_accum/(step + 1), acc.item()

def trainmulti(args, model, loader, optimizer, weights, device):
    """
    train model for GO and EC task
    """
    model.train()
    loss_accum = 0
    acc = []
    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
        batch = batch.to(device)
        function = batch.y
        optimizer.zero_grad()
        lossfn = torch.nn.BCELoss(weight=torch.as_tensor(weights).to(device))
        lossview = torch.nn.BCELoss(weight=torch.as_tensor(weights).to(device), reduction='none')
        pred, o1, moe_loss = model(batch, True)
        loss = lossfn(pred.sigmoid(), function)
        losso1 = lossview(o1.sigmoid(), function).mean(dim=1)
        loss = loss + WEIGHTED_WEIGHT* losso1 + moe_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        loss_accum += loss.item()
        acc.append(fmax(pred.detach().cpu().numpy(), function.detach().cpu().numpy()))

    return loss_accum / (step + 1), np.mean(np.array(acc))


def evaluationmulti(model, loader, device):
    """
    evaluation model for GO and EC task
    """
    model.eval()
    acc = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        function = batch.y.cpu().numpy()
        with torch.no_grad():
            pred, o1, moe_loss = model(batch, False)
        acc.append(fmax(pred.sigmoid().detach().cpu().numpy(), function))

    return np.mean(np.array(acc))

    
def main():
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in Dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    ### Data
    parser.add_argument('--dataset', type=str, default='fold', help='Func or fold')
    parser.add_argument('--dataset_path', type=str, default='Protein', help='path to load and process the data')
    
    parser.add_argument('--mask', action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')

    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=384, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
    parser.add_argument('--fix_dist', action='store_true')

    ### Training hyperparameter
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size')


    parser.add_argument('--lr', default=0.01, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[100, 300], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--checkpoint_to_load', type=str, help='Checkpoint to load')

    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # load datasets
    print('Loading Train & Val & Test Data...')

    if args.dataset == 'func':
        train_set = RCdataset(root=args.dataset_path + '/ProtFunct', split='Train')
        val_set = RCdataset(root=args.dataset_path + '/ProtFunct', split='Val')
        test_set = RCdataset(root=args.dataset_path + '/ProtFunct', split='Test')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        print('Done!')
        print('Train, val, test:', train_set, val_set, test_set)
    elif args.dataset == 'fold':
        train_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='training')
        val_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='validation')
        test_fold = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_fold')
        test_super = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_superfamily')
        test_family = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_family')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_fold_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_super_loader = DataLoader(test_super, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_family_loader = DataLoader(test_family, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test (fold, superfamily, family):', train_set, val_set, test_fold, test_super, test_family)
    elif args.dataset == 'go':
        train_set = GOdataset(root=args.dataset_path + '/GeneOntology', split='train')
        val_set = GOdataset(root=args.dataset_path + '/GeneOntology', split='valid')
        test_set30 = GOdataset(root=args.dataset_path + '/GeneOntology', split='test30', p=30, level="cc")
        test_set40 = GOdataset(root=args.dataset_path + '/GeneOntology', split='test40', p=40, level="cc")
        test_set50 = GOdataset(root=args.dataset_path + '/GeneOntology', split='test50', p=50, level="cc")
        test_set70 = GOdataset(root=args.dataset_path + '/GeneOntology', split='test70', p=70, level="cc")
        test_set95 = GOdataset(root=args.dataset_path + '/GeneOntology', split='test95', p=95, level="cc")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader30 = DataLoader(test_set30, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers)
        test_loader40 = DataLoader(test_set40, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers)
        test_loader50 = DataLoader(test_set50, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers)
        test_loader70 = DataLoader(test_set70, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers)
        test_loader95 = DataLoader(test_set95, batch_size=args.eval_batch_size, shuffle=False,num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test:', train_set, val_set, test_set30, test_set40, test_set50, test_set70, test_set95)
    elif args.dataset == 'ec':
        train_set = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='train')
        val_set = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='valid')
        test_set30 = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='test30')
        test_set40 = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='test40')
        test_set50 = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='test50')
        test_set70 = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='test70')
        test_set95 = ECdataset(root=args.dataset_path + '/EnzymeCommission', split='test95')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader30 = DataLoader(test_set30, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.num_workers)
        test_loader40 = DataLoader(test_set40, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.num_workers)
        test_loader50 = DataLoader(test_set50, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.num_workers)
        test_loader70 = DataLoader(test_set70, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.num_workers)
        test_loader95 = DataLoader(test_set95, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    else:
        print('not supported dataset')
    

    # set up model
    model = MMpgNet()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i - 1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (
                    args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    if args.continue_training:
        checkpoint = torch.load(args.checkpoint_to_load)
        print(f"Continue training, loaded checkpoint from {args.checkpoint_to_load}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        lr_scheduler.last_epoch = 0
        print(f"Continue training from epoch {checkpoint['epoch']}")
        start_epoch = 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    else:
        start_epoch = 1

    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)
    print(f"weighted_weight={WEIGHTED_WEIGHT}")

    if args.dataset == 'func':
        best_val_acc = 0
        test_at_best_val_acc = 0
        
        for epoch in range(start_epoch, args.num_epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            
            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_loss, test_acc = evaluation(args, model, test_loader, device)
            t_end_test = time.perf_counter() 

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_at_best_val_acc = test_acc

                if args.save_checkpoint and (epoch > 1 or args.continue_training):
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_val_acc, args.checkpoint_dir)

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, Test: Loss:{:.6f} Acc:{:.4f}, test_acc@best_val:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, test_at_best_val_acc, t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))

            lr_scheduler.step()

    
    elif args.dataset == 'fold':
        best_val_acc = 0
        test_fold_at_best_val_acc = 0
        test_super_at_best_val_acc = 0
        test_family_at_best_val_acc = 0
        
        for epoch in range(start_epoch, args.num_epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()

            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()

            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_fold_loss, test_fold_acc = evaluation(args, model, test_fold_loader, device)
            test_super_loss, test_super_acc = evaluation(args, model, test_super_loader, device)
            test_family_loss, test_family_acc = evaluation(args, model, test_family_loader, device)
            t_end_test = time.perf_counter()

            if  val_acc > best_val_acc:
                best_val_acc = val_acc    
                test_fold_at_best_val_acc = test_fold_acc
                test_super_at_best_val_acc = test_super_acc
                test_family_at_best_val_acc = test_family_acc

                # if args.save_checkpoint:
                if args.save_checkpoint and (epoch > 1 or args.continue_training):
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_val_acc, args.checkpoint_dir)

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, '\
                'Test_fold: Loss:{:.6f} Acc:{:.4f}, Test_super: Loss:{:.6f} Acc:{:.4f}, Test_family: Loss:{:.6f} Acc:{:.4f}, '\
                'test_fold_acc@best_val:{:.4f}, test_super_acc@best_val:{:.4f}, test_family_acc@best_val:{:.4f}, '\
                'time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, 
                test_fold_loss, test_fold_acc, test_super_loss, test_super_acc, test_family_loss, test_family_acc, 
                test_fold_at_best_val_acc, test_super_at_best_val_acc, test_family_at_best_val_acc, 
                t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))

            lr_scheduler.step()

    elif args.dataset == 'go':
        best_val_acc = 0
        best_30 = 0
        best_40 = 0
        best_50 = 0
        best_70 = 0
        best_95 = 0

        for epoch in range(start_epoch, args.num_epochs + 1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            train_loss, train_acc = trainmulti(args, model, train_loader, optimizer, train_set.weights, device)
            t_end_train = time.perf_counter()
            val_acc = evaluationmulti(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_acc30 = evaluationmulti(args, model, test_loader30, device)
            test_acc40 = evaluationmulti(args, model, test_loader40, device)
            test_acc50 = evaluationmulti(args, model, test_loader50, device)
            test_acc70 = evaluationmulti(args, model, test_loader70, device)
            test_acc95 = evaluationmulti(args, model, test_loader95, device)
            t_end_test = time.perf_counter()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_30 = test_acc30
                best_40 = test_acc40
                best_50 = test_acc50
                best_70 = test_acc70
                best_95 = test_acc95

                if args.save_checkpoint and (epoch > 1 or args.continue_training):
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_val_acc, args.checkpoint_dir)

            t_end = time.perf_counter()
            print(
                'Train: Loss:{:.6f} Acc:{:.4f}, Validation: Acc:{:.4f}, test_acc30@best_val:{:.4f},test_acc40@best_val:{:.4f},test_acc50@best_val:{:.4f},test_acc70@best_val:{:.4f},test_acc95@best_val:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                    train_loss, train_acc, val_acc, best_30, best_40, best_50, best_70, best_95,
                    t_end - t_start, t_end_train - t_start, t_end_test - t_start_test))

            lr_scheduler.step()

    elif args.dataset == 'ec':
        best_val_acc = 0
        best_30 = 0
        best_40 = 0
        best_50 = 0
        best_70 = 0
        best_95 = 0

        for epoch in range(start_epoch, args.num_epochs + 1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            train_loss, train_acc = trainmulti(args, model, train_loader, optimizer, train_set.weights, device)
            t_end_train = time.perf_counter()
            val_acc = evaluationmulti(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_acc30 = evaluationmulti(args, model, test_loader30, device)
            test_acc40 = evaluationmulti(args, model, test_loader40, device)
            test_acc50 = evaluationmulti(args, model, test_loader50, device)
            test_acc70 = evaluationmulti(args, model, test_loader70, device)
            test_acc95 = evaluationmulti(args, model, test_loader95, device)
            t_end_test = time.perf_counter()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_30 = test_acc30
                best_40 = test_acc40
                best_50 = test_acc50
                best_70 = test_acc70
                best_95 = test_acc95

                if args.save_checkpoint and (epoch > 1 or args.continue_training):
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_val_acc, args.checkpoint_dir)

            t_end = time.perf_counter()
            print(
                'Train: Loss:{:.6f} Acc:{:.4f}, Validation: Acc:{:.4f}, test_acc30@best_val:{:.4f},test_acc40@best_val:{:.4f},test_acc50@best_val:{:.4f},test_acc70@best_val:{:.4f},test_acc95@best_val:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                    train_loss, train_acc, val_acc, best_30, best_40, best_50, best_70, best_95,
                    t_end - t_start, t_end_train - t_start, t_end_test - t_start_test))

            lr_scheduler.step()

if __name__ == "__main__":
    main()
