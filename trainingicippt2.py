
import torch 
from regularizers import anscombe_transform, total_time_regularizer, custom_regularizer
# import wandb
import os

def train_model(model, save_path, xrf_params, loader, loss_fn, opt, device, epochs, N, K, t_avg):
    model.train()
    batches = int(torch.ceil(torch.ones(1) * loader.dataset.shape[-2] * loader.dataset.shape[-1] / N).item())
    loss_hist = []
    best_loss = float('inf')
    loss_decrease_tol = 0.005
    # t0 = t0.to(device)

    for epoch in range(1, epochs + 1):
        epoch_losses = []  # Track batch losses within an epoch

        for batch, (crops, t0_crops, ups_crop, p, u) in enumerate(loader, 1):

            crops = crops.to(device)
            ups_crop = ups_crop.to(device)
            t0_crops = t0_crops.to(device)  
            p = p.to(device)
            u = u.to(device)

            # Reformat indices
            # pr = p[:, 0, :].flatten()
            # pc = p[:, 1, :].flatten()
            # ur = u[:, 0, :].flatten()
            # uc = u[:, 1, :].flatten()

            # ii = torch.arange(crops.shape[0] * K) // K
            ans_crops = anscombe_transform(crops)
            input1 = (ans_crops - xrf_params[0]) / xrf_params[1]
            input2 = t0_crops
            t = model(input1, input2).to(device)
            # t = model(input1).to(device)
            # t = model(input1)[ii, :, pr, pc ].to(device)

            # t = model(input1, input2)[ii, :, pr, pc].to(device)

            # Rescale dwell times to the average dwell time
            # alpha = torch.Tensor([t_avg *100])
            # beta = alpha/t_avg
            # gamma_dist = torch.distributions.gamma.Gamma(alpha, beta)

            # t_batch= gamma_dist.rsample([1]).item()
            # t = t * (t_batch / t.mean())
            # xrf_new = (t0 * crops[ii, :, pr, pc] + t * crops[ii, :, ur, uc]) / (t0 + t)
            t = t * (t_avg / t.mean())
            xrf_new = (t0_crops * crops + t * ups_crop) / (t0_crops + t)

            # xrf_new = (t0_crops * crops + t * crops) / (t0_crops + t)

            # TODO: create a dataset of the model output, ups sample and produce output
            
            # Compute loss
            loss = loss_fn(xrf_new, ups_crop).to(device)

            # Track batch loss for this epoch
            epoch_losses.append(loss.item())

            # Backward pass and optimization
            loss.backward()
            opt.step()
            opt.zero_grad()

            # wandb.log({"Batch Loss": loss.item()})
            print(f'Epoch: {epoch:01d}/{epochs} | Batch: {batch:04d}/{batches} | Loss: {loss.item():.05f}')

        # Compute average loss for the epoch and append to loss history
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_hist.append(avg_epoch_loss)
        
        # wandb.log({"Epoch Loss": avg_epoch_loss, "Epoch": epoch})

        # stop if model doesn't learn more upon running 99% of the best loss
        if (best_loss- avg_epoch_loss) < loss_decrease_tol * best_loss: 
            print(f'Early stopping at epoch {epoch}')
            # Save model weights
            weights_dir = "NN_weights"
            os.makedirs(weights_dir, exist_ok=True)
            model_weights_path = os.path.join(weights_dir, save_path)
            torch.save(model.state_dict(), model_weights_path)
            # wandb.save(model_weights_path) 
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
        for crop, t0_crop, r, c in loader:
            crop = crop.to(device)
            t0_crop = t0_crop.to(device)
            r = r.to(device)
            c = c.to(device)
            crop = anscombe_transform(crop)
            input1 = (crop - xrf_parameters[0]) / xrf_parameters[1]
            input2 = t0_crop
            t[0,r,c] = model(input1, input2)[:,0,crop_size//2,crop_size//2]
            # print(model((crop-xrf.mean())/xrf.std()).shape)
            # t[0,r,c] = model((crop-xrf_parameters[0])/xrf_parameters[1])[:,0,crop_size//2,crop_size//2]    
            # print(model((crop-xrf_parameters[0])/xrf_parameters[1]))
            
    t *= t_avg / t.mean()
    print(f'Requested average time: {t_avg}\nReceived average time:  {t.mean().item()}\n')
    print(f'Requested total time: {t_total}\nReceived total time:  {t.sum().item()}')
    return t