from torch.optim import SGD
from time import time
import numpy as np
from matplotlib import pyplot as plt

def save_checkpoint(model, optimizer, epoch, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    print(f"Saved checkpoint to: {path}")

def train_model(model, optimizer, training_dl=None, val_dl=None, test_dl=None, lr=1e-4, start_epoch=0, min_delta=0.1, 
                patience=2, epochs=5, training=True, loading=False, save_path=""):

  '''
  Trains given model with previously declared optimizer, implementing early stopping with patience and a min_delta. 
  Training modes:
  training = True (default) ==> model proceeds to undergo training, saving model parameters, current epoch, and optimizer state after each epoch.
  training = False ==> model will proceed in testing mode, freezing gradients and outputting testing accuracy

  loading modes:
  If loading = False (default) ==> the function will generate a new optimizer and start epochs from 0
  If loading = True ==> the function will use the provided optimizer, which should have been loaded with a previously saved state, 
                        and the provided start_epoch to continue training
 
  '''
  #Checking gpu access
  if torch.cuda.is_available():
    gpu = torch.device("cuda")
  elif torch.backends.mps.is_available():
    gpu = torch.device("mps")
  else:
    gpu = torch.device("cpu") # Fallback to CPU if no GPU available

  model = model.to(gpu)
  loss = nn.CrossEntropyLoss()
  epochs = epochs
  loss_list = []
  accuracy_list = []
  start_time = time()
  epoch_times = []
  counter = 0
  min_delta = min_delta
  patience = patience
  best_accuracy = 0

  if not loading:
    start_epoch = 0
    optimizer = SGD(model.parameters(), lr=lr)
  else: 
    start_epoch = start_epoch

  if training:
    for epoch in range(start_epoch, epochs):
      train_loss = 0
      correct_preds = 0
      epoch_start = time()
      for (inputs, label) in training_dl:
        #Transfer to gpu
        inputs = inputs.to(gpu)
        label = label.to(gpu)
        #Forward Propagation
        outputs = model(inputs)
        epoch_loss = loss(outputs, label)
        train_loss += epoch_loss.item()

        #Backward propagation
        epoch_loss.backward()

        #Gradient Step
        optimizer.step()
        optimizer.zero_grad()
        
      save_checkpoint(model, optimizer, epoch+1, path=save_path)
      loss_list.append(train_loss)
      #Validation
      with torch.no_grad():
        for (inputs, label) in val_dl:
          inputs = inputs.to(gpu)
          label = label.to(gpu)
          outputs = model(inputs)
          correct_preds += (torch.argmax(outputs, dim=1) == label).sum().item()

      accuracy = correct_preds / len(val_loader.dataset)
      accuracy_list.append(accuracy)
      epoch_end = time()
      epoch_times.append(epoch_end-epoch_start)
      print(f'epoch: {epoch+1}, training loss: {train_loss}, accuracy:{accuracy}')

      #Early stopping
      if accuracy > best_accuracy:
        best_accuracy = accuracy
      if accuracy < best_accuracy - min_delta:
        counter += 1
        if counter >= patience:
          print(f'Early stopping at epoch {epoch+1}')
          break
      else:
        counter = 0

    end_time = time()
    total_training_time = end_time - start_time
    avg_time_epoch = np.mean(epoch_times)
    print(f'Total trianing time: {total_training_time}')
    print(f'Average time per epoch: {avg_time_epoch}')
    
    return loss_list, accuracy_list, avg_time_epoch
  else:
    with torch.no_grad():
      correct_preds = 0
      for (inputs, label) in test_dl:
        inputs = inputs.to(gpu)
        label = label.to(gpu)
        outputs = model(inputs)
        correct_preds += (torch.argmax(outputs, dim=1) == label).sum().item()

    accuracy = correct_preds / len(test_dl.dataset)
    print(f'accuracy:{accuracy}')


def plot_training_curves(train_losses, val_accuracies):
    '''
    Helper function to visualize performance during training
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(val_accuracies)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_ablation_training_curves(train_losses, val_accuracies,
                                  learning_rates, avg_time_epoch):
    '''
    Helper function to visualize performance during training
    '''

    for i in range(len(learning_rates)):

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

      ax1.plot(train_losses[i])
      ax1.set_title(f'Training Loss, lr = {learning_rates[i]}')
      ax1.set_xlabel('Epoch')
      ax1.set_ylabel('Loss')
      ax1.grid(True)

      ax2.plot(val_accuracies[i])
      ax2.set_title(f'Accuracy, lf = {learning_rates[i]}')
      ax2.set_xlabel('Epoch')
      ax2.set_ylabel('Accuracy')
      ax2.grid(True)

      plt.tight_layout()
      plt.show()



def embeddings_cashing(dir, train_ids):

  embedding_cache = {}

  i = 0
  for img_id in train_ids:
      img_info = coco.loadImgs(img_id)[0]
      path = os.path.join("/content/livecell/images/images/livecell_train_val_images", img_info['file_name'])

      image = cv2.imread(path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      input_image = transform.apply_image(image)
      input_image = torch.as_tensor(input_image, device=device)
      input_image = input_image.permute(2,0,1).unsqueeze(0)
      input_image = sam.preprocess(input_image)

      with torch.no_grad():
          embedding = sam.image_encoder(input_image)

      embedding_cache[img_id] = embedding.cpu()  # store on CPU
      i += 1
      if i % 100 == 0:
          print(f"Image {i}/{len(train_ids)} stored")
  return embedding_cache


def train_one_epoch_sam(embedding_cache, train_loader):
    sam.train()
    total_loss = 0
    i = 0

    for images, masks, boxes in train_loader:

        for image, gt_mask, box in zip(images, masks, boxes):

            image_embedding = embedding_cache[img_id].to(device)

            # --- box prompt ---
            box_t = transform.apply_boxes(box[None, :], image.shape[:2])
            box_t = torch.as_tensor(box_t, dtype=torch.float, device=device)

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=box_t,
                masks=None,
            )

            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # --- GT mask ---
            gt_mask_torch = torch.tensor(gt_mask, device=device).float()
            gt_mask_torch = gt_mask_torch.unsqueeze(0).unsqueeze(0)

            gt_mask_torch = F.interpolate(
                gt_mask_torch,
                size=low_res_masks.shape[-2:],
                mode="nearest"
            )

            # --- loss ---
            loss = loss_fn(low_res_masks, gt_mask_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss


def validate_sam(embedding_cache, val_loader):
    sam.eval()
    total_loss = 0
    i = 0

    with torch.no_grad():
        for images, masks, boxes in val_loader:

            for image, gt_mask, box in zip(images, masks, boxes):

                image_embedding = embedding_cache[img_id].to(device)

                # --- box prompt ---

                box_t = transform.apply_boxes(box[None, :], image.shape[:2])
                box_t = torch.as_tensor(box_t, dtype=torch.float, device=device)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_t,
                    masks=None,
                )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                gt_mask_torch = torch.tensor(gt_mask, device=device).float()
                gt_mask_torch = gt_mask_torch.unsqueeze(0).unsqueeze(0)

                gt_mask_torch = F.interpolate(
                    gt_mask_torch,
                    size=low_res_masks.shape[-2:],
                    mode="nearest"
                )

                loss = loss_fn(low_res_masks, gt_mask_torch)
                total_loss += loss.item()

    return total_loss
