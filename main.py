from AutoEncoder.train_autoencoder import train_auto_encoder
from Classifier.train_classifier import train_model_normal, train_model_sam, eval_model


autoencoder_num_epochs = 100
autoencoder_lr = 0.001

classifier_num_epochs = 20
classifier_lr = 0.001
sam_classifier_lr = 0.001

train_autoencoder = True

classifier_optim_mode = "sam" # "normal"  normal mode uses Adam optimizer 

if train_autoencoder:
    print("This is pre-training step for autoencoder... the encoder weights will be utilized in Classifier...\n")
    print("Beginning training autoencoder...")
    train_auto_encoder(lr=autoencoder_lr, num_epochs=autoencoder_num_epochs)

if classifier_optim_mode == "sam":
    train_model_sam(lr= sam_classifier_lr, num_epochs=classifier_num_epochs)
    print("Training model with SAM optimizer finished... Evaluating model...")
    eval_model(classifier_optim_mode)
elif classifier_optim_mode == "normal":
    train_model_normal(lr=classifier_lr, num_epochs=classifier_num_epochs)
    print("Training model with Normal optimizer finished... Evaluating model...")
    eval_model(classifier_optim_mode)
else:
    raise ValueError("The value for classifier_optim_mode should be either sam or normal")
