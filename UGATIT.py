import os
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
import time
from dataset import reader_creater
#from network import ResnetGenerator, Discriminator, clip_rho
from networks import *
from utils import RGB2BGR, tensor2numpy, cam, denorm
from glob import glob
from visualdl import LogWriter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('train.log')
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Hello!!!')

class UGATIT(object):
    def __init__(self, args):
        self.light = args.light
        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.device = args.device
        self.phase = args.phase

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.place = fluid.CUDAPlace(2) if self.device == 'cuda' else fluid.CPUPlace()
        self.start_iter = 1

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ DataLoader """
        self.trainA = reader_creater(os.path.join('dataset', self.dataset, 'trainA'), shuffle=True)
        self.trainB = reader_creater(os.path.join('dataset', self.dataset, 'trainB'), shuffle=True)
        if self.phase == 'train':
            self.testA = reader_creater(os.path.join('dataset', self.dataset, 'testA'))
            self.testB = reader_creater(os.path.join('dataset', self.dataset, 'testB'))
        if self.phase == 'test':
            self.testA = reader_creater(os.path.join('dataset', self.dataset, 'testA'),cycle=False)
            self.testB = reader_creater(os.path.join('dataset', self.dataset, 'testB'),cycle=False)

        
        self.trainA_loader = paddle.batch(self.trainA, self.batch_size)()
        self.trainB_loader = paddle.batch(self.trainB, self.batch_size)()
        self.testA_loader = paddle.batch(self.testA, 1)()
        self.testB_loader = paddle.batch(self.testB, 1)()


        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = fluid.dygraph.L1Loss()
        self.MSE_loss = fluid.dygraph.MSELoss()
        self.BCE_loss = fluid.dygraph.BCELoss()
        

        """ Trainer """
        self.G_optim = fluid.optimizer.AdamOptimizer(
            parameter_list=self.genA2B.parameters()+self.genB2A.parameters(), 
            learning_rate= LinearDecay(lr=self.lr, iteration=self.iteration) if self.decay_flag == 1 else self.lr, 
            beta1=0.5, 
            beta2=0.999, 
            regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
            )
        
        self.D_optim = fluid.optimizer.AdamOptimizer(
            parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters(), 
            learning_rate=LinearDecay(lr=self.lr, iteration=self.iteration) if self.decay_flag == 1 else self.lr, 
            beta1=0.5, 
            beta2=0.999, 
            regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
            )  

        print(' [*] Model build SUCCESS')
    
    
    def clear_gradients(self):
        self.genA2B.clear_gradients()
        self.genB2A.clear_gradients()
        self.disGA.clear_gradients()
        self.disGB.clear_gradients()
        self.disLA.clear_gradients()
        self.disLB.clear_gradients()


    def train(self):

        with fluid.dygraph.guard(self.place):                
            
            self.build_model()

            self.genA2B.train()
            self.genB2A.train()
            self.disGA.train()
            self.disGB.train()
            self.disLA.train()
            self.disLB.train()

            if self.resume:
                model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*'))
                if not len(model_list) == 0:
                    model_list.sort()
                    self.start_iter = int(model_list[-1].split('/')[-1])
                print(' [*] Resume training from step %d'%(self.start_iter))
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), self.start_iter)
                print(" [*] Load SUCCESS")
            
            # training loop
            print('Training start !')
            start_time = time.time()
            with LogWriter(logdir="./log") as writer:
                for step in range(self.start_iter, self.iteration + 1):  
                    
                    real_A = next(self.trainA_loader)
                    real_B = next(self.trainB_loader)
                    
                    real_A = np.array(
                        [real_A[0].reshape(3, 256, 256)]).astype("float32")
                    real_B = np.array(
                        [real_B[0].reshape(3, 256, 256)]).astype("float32")

                    real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)
                    
                    # Update D
                    fake_A2B, _, _ = self.genA2B(real_A)
                    fake_B2A, _, _ = self.genB2A(real_B)

                    real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                    real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                    real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                    real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                    fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                    fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                    fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                    fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                    D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, fluid.layers.zeros_like(fake_GA_logit))
                    D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, fluid.layers.zeros_like(fake_GA_cam_logit))
                    D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
                    D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, fluid.layers.zeros_like(fake_LA_cam_logit))
                    D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, fluid.layers.zeros_like(fake_GB_logit))
                    D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, fluid.layers.zeros_like(fake_GB_cam_logit))
                    D_ad_loss_LB = self.MSE_loss(real_LB_logit, fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, fluid.layers.zeros_like(fake_LB_logit))
                    D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, fluid.layers.zeros_like(fake_LB_cam_logit))

                    D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                    D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                    Discriminator_loss = D_loss_A + D_loss_B
                    Discriminator_loss.backward()
                   
                    self.D_optim.minimize(Discriminator_loss)
                    self.clear_gradients()

                    # Update G
                    fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                    fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                    fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                    fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                    fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                    fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                    fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                    fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                    fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                    fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                    G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.layers.ones_like(fake_GA_logit))
                    G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.layers.ones_like(fake_GA_cam_logit))
                    G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.layers.ones_like(fake_LA_logit))
                    G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.layers.ones_like(fake_LA_cam_logit))
                    G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.layers.ones_like(fake_GB_logit))
                    G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.layers.ones_like(fake_GB_cam_logit))
                    G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.layers.ones_like(fake_LB_logit))
                    G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.layers.ones_like(fake_LB_cam_logit))
                    
                    G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                    G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                    G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                    G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
                    
                    G_cam_loss_A = self.BCE_loss(fluid.layers.sigmoid(fake_B2A_cam_logit), fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(fake_A2A_cam_logit), fluid.layers.zeros_like(fake_A2A_cam_logit))
                    G_cam_loss_B = self.BCE_loss(fluid.layers.sigmoid(fake_A2B_cam_logit), fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(fake_B2B_cam_logit), fluid.layers.zeros_like(fake_B2B_cam_logit))
                    
        
                    G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                    G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                    Generator_loss = G_loss_A + G_loss_B
                    Generator_loss.backward()
                                       
                    self.G_optim.minimize(Generator_loss)
                    self.clear_gradients()

                    # clip parameter of AdaILN and ILN, applied after optimizer step
                    def clip_rho(net, vmin=0, vmax=1):
                        for name, param in net.named_parameters():
                            if 'rho' in name:
                                param.set_value(fluid.layers.clip(param, vmin, vmax))                       
                    clip_rho(self.genA2B, 0, 1)
                    clip_rho(self.genB2A, 0, 1)

                    print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                    logger.info("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                    writer.add_scalar(tag="D_loss_A/D_ad_loss_GA", step=step, value=D_ad_loss_GA.numpy())
                    writer.add_scalar(tag="D_loss_A/D_ad_cam_loss_GA", step=step, value=D_ad_cam_loss_GA.numpy())
                    writer.add_scalar(tag="D_loss_A/D_ad_loss_LA", step=step, value=D_ad_loss_LA.numpy())
                    writer.add_scalar(tag="D_loss_A/D_ad_cam_loss_LA", step=step, value=D_ad_cam_loss_LA.numpy())

                    writer.add_scalar(tag="D_loss_B/D_ad_loss_GB", step=step, value=D_ad_loss_GB.numpy())
                    writer.add_scalar(tag="D_loss_B/D_ad_cam_loss_GB", step=step, value=D_ad_cam_loss_GB.numpy())
                    writer.add_scalar(tag="D_loss_B/D_ad_loss_LB", step=step, value=D_ad_loss_LB.numpy())
                    writer.add_scalar(tag="D_loss_B/D_ad_cam_loss_LB", step=step, value=D_ad_cam_loss_LB.numpy())
                    
                    writer.add_scalar(tag="D_loss_A/D_loss_A", step=step, value=D_loss_A.numpy())
                    writer.add_scalar(tag="D_loss_B/D_loss_B", step=step, value=D_loss_B.numpy())

                    writer.add_scalar(tag="G_loss_A/G_ad_loss_GA", step=step, value=G_ad_loss_GA.numpy())
                    writer.add_scalar(tag="G_loss_A/G_ad_cam_loss_GA", step=step, value=G_ad_cam_loss_GA.numpy())
                    writer.add_scalar(tag="G_loss_A/G_ad_loss_LA", step=step, value=G_ad_loss_LA.numpy())
                    writer.add_scalar(tag="G_loss_A/G_ad_cam_loss_LA", step=step, value=G_ad_cam_loss_LA.numpy())

                    writer.add_scalar(tag="G_loss_B/G_ad_loss_GB", step=step, value=G_ad_loss_GB.numpy())
                    writer.add_scalar(tag="G_loss_B/G_ad_cam_loss_GB", step=step, value=G_ad_cam_loss_GB.numpy())
                    writer.add_scalar(tag="G_loss_B/G_ad_loss_LB", step=step, value=G_ad_loss_LB.numpy())
                    writer.add_scalar(tag="G_loss_B/G_ad_cam_loss_LB", step=step, value=G_ad_cam_loss_LB.numpy())
                    
                    writer.add_scalar(tag="G_loss_A/G_loss_A", step=step, value=G_loss_A.numpy())
                    writer.add_scalar(tag="G_loss_B/G_loss_B", step=step, value=G_loss_B.numpy())
                    
                    writer.add_scalar(tag="G_loss_A/G_recon_loss_A", step=step, value=G_recon_loss_A.numpy())
                    writer.add_scalar(tag="G_loss_B/G_recon_loss_B", step=step, value=G_recon_loss_B.numpy())
                    writer.add_scalar(tag="G_loss_A/G_identity_loss_A", step=step, value=G_identity_loss_A.numpy())
                    writer.add_scalar(tag="G_loss_B/G_identity_loss_B", step=step, value=G_identity_loss_B.numpy())


                    writer.add_scalar(tag="Loss/G_loss", step=step, value=Generator_loss.numpy())
                    writer.add_scalar(tag="Loss/D_loss", step=step, value=Discriminator_loss.numpy())

                    writer.add_scalar(tag="Learning_rate/G_optim.learning_rate", step=step, value=self.G_optim.current_step_lr())
                    writer.add_scalar(tag="Learning_rate/D_optim.learning_rate", step=step, value=self.D_optim.current_step_lr())
                    
                    if step % self.print_freq == 0:
                        train_sample_num = 5
                        test_sample_num = 5
                        A2B = np.zeros((self.img_size * 7, 0, 3))
                        B2A = np.zeros((self.img_size * 7, 0, 3))

                        self.genA2B.eval()
                        self.genB2A.eval()
                        self.disGA.eval()
                        self.disGB.eval()
                        self.disLA.eval()
                        self.disLB.eval()
                        
                        for _ in range(train_sample_num):
                            real_A = next(self.trainA_loader)
                            real_B = next(self.trainB_loader)
                            real_A = np.array(
                                [real_A[0].reshape(3, 256, 256)]).astype("float32")
                            real_B = np.array(
                                [real_B[0].reshape(3, 256, 256)]).astype("float32")

                            real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                            A2B = np.concatenate((A2B, 
                                np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 
                                0)), 
                            1)
                            
                            B2A = np.concatenate((B2A, 
                                np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 
                                0)), 
                            1)
                            
                        for _ in range(test_sample_num):
                            
                            real_A = next(self.testA_loader)
                            real_B = next(self.testB_loader)
                            real_A = np.array(
                                [real_A[0].reshape(3, 256, 256)]).astype("float32")
                            real_B = np.array(
                                [real_B[0].reshape(3, 256, 256)]).astype("float32")

                            real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                            A2B = np.concatenate((A2B, 
                                np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 
                                0)), 
                            1)

                            B2A = np.concatenate((B2A, 
                                np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 
                                0)), 
                            1)
                            
                        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                        cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                        
                        self.genA2B.train()
                        self.genB2A.train()
                        self.disGA.train()
                        self.disGB.train()
                        self.disLA.train()
                        self.disLB.train()

                    if step % self.save_freq == 0:
                        self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
                    
    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        print(' [*] Saving parameter in %d/%d... ...'%(step, self.iteration))
        print(' [#] save model parameters...')
        for i in params:
            fluid.dygraph.save_dygraph(params[i], os.path.join(dir, '%07d'%step ,'%s'%i + '_%07d'%step))
        print(' [#] save optimizer parameters...')
        fluid.dygraph.save_dygraph(self.G_optim.state_dict(), os.path.join(dir, '%07d'%step ,'genA2B' + '_%07d'%step))
        fluid.dygraph.save_dygraph(self.D_optim.state_dict(), os.path.join(dir, '%07d'%step ,'disGA' + '_%07d'%step))
        print('Save done!')
        


    def load(self, dir, step):
        load_path = os.path.join(dir, '%07d'%step)
        if self.phase == 'train':
            genA2B, g_optim = fluid.dygraph.load_dygraph(os.path.join(load_path, 'genA2B_%07d'%step))
            self.genA2B.set_dict(genA2B)
            self.G_optim.set_dict(g_optim)
            genB2A, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'genB2A_%07d'%step))
            self.genB2A.set_dict(genB2A)
            disGA, d_optim = fluid.dygraph.load_dygraph(os.path.join(load_path, 'disGA_%07d'%step))
            self.disGA.set_dict(disGA)
            self.D_optim.set_dict(d_optim)
            disGB, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'disGB_%07d'%step))
            self.disGB.set_dict(disGB)
            disLA, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'disLA_%07d'%step))
            self.disLA.set_dict(disLA)
            disLB, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'disLB_%07d'%step))
            self.disLB.set_dict(disLB)
        else:
            genA2B, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'genA2B_%07d'%step))
            self.genA2B.set_dict(genA2B)
            genB2A, _ = fluid.dygraph.load_dygraph(os.path.join(load_path, 'genB2A_%07d'%step))
            self.genB2A.set_dict(genB2A)


    def test(self):
        
        with fluid.dygraph.guard(self.place): 
            self.build_model()

            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('/')[-1])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return
                    
            self.genA2B.eval()
            self.genB2A.eval()

            
            for n, real_A in enumerate(self.testA_loader):
                real_A = np.array(
                    [real_A[0].reshape(3, 256, 256)]).astype("float32")
                real_A = fluid.dygraph.to_variable(real_A)

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A) 

                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)
                
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

            
            for n, real_B in enumerate(self.testB_loader):
                real_B = np.array(
                    [real_B[0].reshape(3, 256, 256)]).astype("float32")
                real_B = fluid.dygraph.to_variable(real_B)

                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

                
class LinearDecay(paddle.fluid.dygraph.learning_rate_scheduler.LearningRateDecay):
    def __init__(self, lr, iteration):
        super(LinearDecay, self).__init__()
        self.lr = lr
        self.iteration = iteration

    def step(self):
        current_lr = self.lr      
        if self.step_num > (self.iteration // 2):
            current_lr -= (self.step_num - (self.iteration // 2)) * (self.lr) / (self.iteration // 2)
        return self.create_lr_var(current_lr)
