b=[]
Loss = []
lable = []

my_net.eval()
with torch.no_grad():
    for data in test_loader:
        predict_value,zero,ceshi,V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,gate,guance,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = my_net(data,A,device)  # [B, N, 1, D]
        b.append(predict_value)
        lable.append(data["flow_y"])
        for shijian in range(10):      
            loss_RMSE = criterion(predict_value[:,:,shijian], data["flow_y"][:,:,shijian])
            loss_MAE = nn.L1Loss()(predict_value[:,:,shijian], data["flow_y"][:,:,shijian])
            Loss.append(loss_RMSE)
            print("Test RMSE: {:02.8f},Test MAE: {:02.8f}".format(loss_RMSE.sqrt(),loss_MAE)) 
