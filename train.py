os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
my_net = ChebNet()
device = torch.device("cuda")
my_net = my_net.to(device)
criterion = nn.MSELoss()
train_loss = []
val_loss = []
test_loss = []
optimizer = optim.Adam(params=my_net.parameters(),lr=0.0003)
Epoch = 3000
min_loss = 100000
import torch.nn.functional as F
my_net.train()
for epoch in range(Epoch):
    my_net.train()
    for data in train_loader:# ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
        my_net.zero_grad( )
        predict_value,zero,ceshi,V,gate,res_1,res_2,res_3,res_4,res_5,res_6,res_7,res_8,step_0,step_1 = my_net(data,A,device)
              
        
        loss = criterion(predict_value, data["flow_y"])+criterion(zero, ceshi)+criterion(zero,V+gate)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    if epoch%100==0:
        torch.save(my_net,'./model_FGCNDE_PEMS04_ceshi'+'.pth')
        print("Epoch: {:04d},train_loss: {:02.8f}".format(epoch,loss)) 