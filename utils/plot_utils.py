import matplotlib.pyplot as plt
import torch

def visualizeLeads_comp(data, text, original_ecg, path=None):
  
  fig, axs = plt.subplots(8, figsize=(4*15, 4*8*2.5))
  fig.suptitle(f'{text}', fontsize=50, y=0.92)
  pad_size = (original_ecg.shape[-1] - data.shape[-1])//2
  if pad_size != 0:
      pad = torch.full((8, pad_size), float('nan'))
      data = torch.cat((pad, data, pad), dim=1)
  
  for lead in range(8):
        y = list(data[lead, :])
        axs[lead].plot(list(range(data.shape[-1])),y,linewidth=2, color='red')
        

        y = list(original_ecg[lead, :])
        axs[lead].plot(list(range(original_ecg.shape[-1])), y, linewidth=1, color='blue')
        axs[lead].set_xlabel(f'Lead {lead}',fontsize=30)
        axs[lead].xaxis.label.set_visible(True)

        axs[lead].tick_params(axis='x', labelsize=30)  # For x-axis
        axs[lead].tick_params(axis='y', labelsize=30)  # For y-axis
      

  plt.subplots_adjust(hspace=0.4, wspace=0.2)
  plt.savefig(path)
  
  return fig
  
  
  

def plot_samples(images, path, kv=None):
    # images is of shape n, 1, 8, 2500
    visualizeLeads_comp(images[0][0], "Sampled ECG", images[1][0], path)
    return plot_counterfactual_comparison(images=images, count=2, kv=kv)
    

def plot_counterfactual_comparison_vertical(images, counterfactual=None, count=4, kv=None):
  if counterfactual is not None:
    modifications = images - counterfactual
  if kv is not None:
        var = kv['key']
        values = kv['values']
        
  # Create a figure with count*8 rows (with extra space between ECGs)
  fig, axes = plt.subplots(count*8, 1, figsize=(15, 10*count + 10))
  plt.subplots_adjust(hspace=0.5)  # Overall spacing
  
  for i in range(count):
      # Add a title for each ECG group
      if kv is not None:
          fig.text(0.5, 1 - (i*(8/count))/count - 0.02, f'ECG {i+1} - D {var}, {values[i].item()}', 
                  ha='center', fontsize=12, fontweight='bold')
      
      for lead in range(8):
          ax_idx = i*8 + lead
          axes[ax_idx].set_title(f'Lead {lead}', fontsize=9)
          axes[ax_idx].plot(images[i, 0, lead,:].detach().clone().squeeze().cpu().numpy(), 'k', linewidth=1, linestyle='--')
          if counterfactual is not None:
            axes[ax_idx].plot(modifications[i, 0, lead, :].detach().clone().squeeze().cpu().numpy(), 'r', linewidth=1, linestyle='--')
            axes[ax_idx].plot(counterfactual[i, 0, lead, :].detach().clone().squeeze().cpu().numpy(),'g', linewidth=2)
          
          # Remove extra ticks to reduce clutter
          axes[ax_idx].tick_params(axis='both', labelsize=8)
          
          # Add extra space after each ECG (8 leads)
          if lead == 7 and i < count-1:
              axes[ax_idx].set_xlabel("", fontsize=0)
              axes[ax_idx].xaxis.set_tick_params(labelsize=0)
              axes[ax_idx].spines['bottom'].set_visible(False)
              plt.subplots_adjust(hspace=1.0)  # Extra space between ECG groups
  
  plt.tight_layout()
  return fig
 
    
def plot_counterfactual_comparison(images, counterfactual=None, count=4, kv=None):
  if counterfactual is not None:
    modifications = images - counterfactual
  if kv is not None:
        var = kv['key']
        values = kv['values']
  subDims = [8, count]
  fig, axes = plt.subplots(subDims[0], subDims[1], figsize=(15, 10))
  plt.subplots_adjust(hspace=0.4, wspace=0.3)
  for i in range(count):
      
      for lead in range(8):
          axes[lead, i].title.set_text(f'D {var}, {values[i].item()}')
          axes[lead, i].plot(images[i, 0, lead,:].detach().clone().squeeze().cpu().numpy(), 'k', linewidth=1, linestyle='--')
          if counterfactual is not None:
            axes[lead, i].plot(modifications[i, 0, lead, :].detach().clone().squeeze().cpu().numpy(), 'r', linewidth=1, linestyle='--')
            axes[lead, i].plot(counterfactual[i, 0, lead, :].detach().clone().squeeze().cpu().numpy(),'g', linewidth=2)
  return fig