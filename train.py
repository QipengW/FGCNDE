os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
my_net = ChebNet()
device = torch.device("cuda")
my_net = my_net.to(device)
criterion = nn.MSELoss()
train_loss = []
val_loss = []
test_loss = []
optimizer = optim.Adam(params=my_net.parameters(),lr=0.0001,weight_decay=0.000001)
Epoch = 3000
import torch.nn.functional as F
my_net.train()
for epoch in range(Epoch):
    my_net.train()
    for data in train_loader:# ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
        my_net.zero_grad( )
        pre,zero,ceshi,V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,gate,res_1,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = my_net(data,device)
        loss = criterion(pre, data["flow_y"])+0.01*criterion(zero, ceshi)+0.01*(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12
                                                                                          +V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    if epoch%100==0:
        torch.save(my_net,'./model_SFGDE'+'.pth')
        print("Epoch: {:04d},train_loss: {:02.8f}".format(epoch,loss)) 
