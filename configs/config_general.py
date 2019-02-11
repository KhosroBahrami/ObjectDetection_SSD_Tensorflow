


# Backbone network for SSD:
# Uncomment one of the model to use for SSD object detection for train, test & demo.

backbone_network = 'ssd_vgg_300'
#backbone_network = 'ssd_mobilenet_v1'
#backbone_network = 'ssd_mobilenet_v2'
#backbone_network = 'ssd_resnet_v1'
#backbone_network = 'ssd_resnet_v2'
#backbone_network = 'ssd_inception_v4'
#backbone_network = 'ssd_inception_resnet_v2'




# Mode of training
# Uncomment one of the traning modes for traning of SSD object detection 
 
# Fine tuning the existing SSD model (based on a backbone network, such as MobileNet, ResNet, ...)
training_mode = 'finetune_on_existing_ssd_model'

# Fine tuning a new SSD model (based on a backbone network, such as MobileNet, ResNet, ...)
#training_mode = 'finetune_on_new_ssd_model_step1'  

# Fine tuning the new SSD model (based on a backbone network, such as MobileNet, ResNet, ...)
#training_mode = 'finetune_on_new_ssd_model_step2'  
