
import torch 
from regularizers import anscombe_transform, total_time_regularizer, custom_regularizer

def train_model(model, xrf_params, loader, loss_fn, opt, device, epochs, N, K, t_avg, t0, beta, target_total_time):
    model.train()
    batches = int(torch.ceil(torch.ones(1) * loader.dataset.shape[-2] * loader.dataset.shape[-1] / N).item())
    loss_hist = []
    best_loss = float('inf')
    loss_decrease_tol = 0.005

    for epoch in range(1, epochs + 1):
        epoch_losses = []  # Track batch losses within an epoch

        for batch, (crops, p, u) in enumerate(loader, 1):
            crops = crops.to(device)
            p = p.to(device)
            u = u.to(device)
            # print(f"crops shape: {crops.shape}, p shape: {p.shape}, u shape: {u.shape}")

            print(f"crops shape: {crops.shape}")
            # print(f"p shape: {p.shape}")
            # print(f"u shape: {u.shape}")

            # Reformat indices
            pr = p[:, 0, :].flatten()
            pc = p[:, 1, :].flatten()
            ur = u[:, 0, :].flatten()
            uc = u[:, 1, :].flatten()

            print(f"pr shape: {pr.shape}")
            print(f"pc shape: {pc.shape}")
            # print(f"ur shape: {ur.shape}")
            # # print(f"uc shape: {uc.shape}")

            ii = torch.arange(crops.shape[0] * K) // K
            ans_crops = anscombe_transform(crops)
            # print(f"ans_crops shape: {ans_crops.shape}")
            t = model((ans_crops - xrf_params[0]) / xrf_params[1])[ii, :, pr, pc].to(device)
            # print("t shape", t.shape)
            # Rescale dwell times to the average dwell time
            # alpha = torch.Tensor([t_avg *100])
            # beta = alpha/t_avg
            # gamma_dist = torch.distributions.gamma.Gamma(alpha, beta)

            # t_batch= gamma_dist.rsample([1]).item()
            # t = t * (t_batch / t.mean())
            # xrf_new = (t0 * crops[ii, :, pr, pc] + t * crops[ii, :, ur, uc]) / (t0 + t)
            t = t * (t_avg / t.mean())
            # Sample
            xrf_new = (t0 * crops[ii,:,pr,pc] + t * crops[ii,:,ur,uc]) / (t0 + t)
            
            # Compute loss
            loss = loss_fn(xrf_new, crops[ii, :, ur, uc]).to(device)
            # time_reg_loss = total_time_regularizer(t, target_total_time, beta=beta)
            # loss = main_loss + time_reg_loss

            # Track batch loss for this epoch
            epoch_losses.append(loss.item())

            # Backward pass and optimization
            loss.backward()
            opt.step()
            opt.zero_grad()

            print(f'Epoch: {epoch:01d}/{epochs} | Batch: {batch:04d}/{batches} | Loss: {loss.item():.05f}')

        # Compute average loss for the epoch and append to loss history
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_hist.append(avg_epoch_loss)
        # stop if model doesn't learn more upon running 99% of the best loss
        if (best_loss- avg_epoch_loss) < loss_decrease_tol * best_loss: 
            print(f'Early stopping at epoch {epoch}')
            break
        else: 
            print(f"epoch improvement: {best_loss- avg_epoch_loss}")
            print(f"best loss {loss_decrease_tol*100}% margin: {best_loss*loss_decrease_tol}")
            best_loss = avg_epoch_loss

        print(f'Epoch {epoch:01d} average loss: {avg_epoch_loss:.05f}')
    return loss_hist, model

def eval_model(model,xrf, loader,xrf_parameters, crop_size, t_avg, t_total, info,device):
    element, loss_func = info
    print(f"Evaluating model for element {element} using {loss_func} loss")
    model.eval()
    t = torch.zeros(1,*xrf.shape[-2:]).to(device)
    with torch.no_grad():
        for crop, r, c in loader:
            crop = crop.to(device)
            r = r.to(device)
            c = c.to(device)
            crop = anscombe_transform(crop)
            # print(model((crop-xrf.mean())/xrf.std()).shape)
            t[0,r,c] = model((crop-xrf_parameters[0])/xrf_parameters[1])[:,0,crop_size//2,crop_size//2]    
            # print(model((crop-xrf_parameters[0])/xrf_parameters[1]))
            
    t *= t_avg / t.mean()
    print(f'Requested average time: {t_avg}\nReceived average time:  {t.mean().item()}\n')
    print(f'Requested total time: {t_total}\nReceived total time:  {t.sum().item()}')
    return t

# def train_model(model, xrf_params, loader, loss_fn, opt, device, epochs, N, K, t_avg, t0, beta, target_total_time):
#     """
#     Function to train a model with different loss functions.

#     Args:
#         model (torch.nn.Module): The neural network model to be trained.
#         loader (torch.utils.data.DataLoader): The data loader for the dataset.
#         loss_fn (torch.nn.Module): The loss function to use.
#         opt (torch.optim.Optimizer): The optimizer.
#         device (torch.device): The device to use for computation (CPU/GPU).
#         epochs (int): The number of training epochs.
#         N (int): Batch size.
#         K (int): Number of pixel masks.
#         t_avg (float): Average dwell time.
#         t0 (float): Initial dwell time.

#     Returns:
#         list: History of the losses per batch during training.
#     """ 
#     model.train()
#     batches = int(torch.ceil(torch.ones(1) * loader.dataset.shape[-2] * loader.dataset.shape[-1] / N).item())
#     loss_hist = []
#     batch = 1

#     for epoch in range(1, epochs + 1):
#         for crops, p, u in loader: 
#             crops = crops.to(device)
#                 # print(crops.min())
#             p = p.to(device)
#             u = u.to(device)
            

#             # Reformat indices
#             pr = p[:,0,:].flatten()
#             pc = p[:,1,:].flatten()
#             ur = u[:,0,:].flatten()
#             uc = u[:,1,:].flatten()
#             # print(model(crops).shape)
#             # print(pr.shape, pc.shape)

#             # Get dwell time
#             # to acess abtch dimensions
#             ii = torch.arange(crops.shape[0] * K) // K
#             ans_crops = anscombe_transform(crops)
#             # scale the input anscombe rates of the model by the overall anscombe mean and std dev for selected 
#             # pixels in the batch
#             t = model((ans_crops-xrf_params[0])/xrf_params[1])[ii,:,pr,pc].to(device)
#             print(f"Min in model output: {t.min().item()}")
#             # print average 
#             print(f"Average in model output: {t.mean().item()}")
#             # print max
#             print("Max in model output", t.max().item())
            
#             # rescale to the average dwell time
#             t = t * (t_avg / t.mean())
            
#             # Sample
#             xrf_new = (t0 * crops[ii,:,pr,pc] + t * crops[ii,:,ur,uc]) / (t0 + t)
            
#             # Compute loss
#             main_loss = loss_fn(xrf_new, crops[ii,:,ur,uc]).to(device)
#             # add on a regularizer 
#             time_reg_loss = total_time_regularizer(t, target_total_time, beta=beta)
#             loss = main_loss + time_reg_loss
#             loss_hist.append(loss.item())
#             # Print progress|
#             print(f'Epoch: {epoch:01d}/{epochs} | Batch: {batch:04d}/{batches} | Loss: {loss.item():.05f}')
            
#             # Backward pass and optimization
#             loss.backward()
#             # optimizer step is called after the backward pass
#             opt.step()
#             opt.zero_grad()
#             batch += 1
#     return loss_hist, model