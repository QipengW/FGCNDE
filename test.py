b=[]
Loss = []
lable = []

my_net.eval()
with torch.no_grad():
    for data in test_loader:
        predict_value,zero,ceshi,V,gate,res_1,res_2,res_3,res_4,res_5,res_6,res_7,res_8,step_0,step_1,output,graph_data,L = my_net(data,A,device)  # [B, N, 1, D]
        b.append(predict_value)
        lable.append(data["flow_y"])
        for shijian in range(10):      
            loss_RMSE = criterion(predict_value[:,:,shijian], data["flow_y"][:,:,shijian])
            loss_MAE = nn.L1Loss()(predict_value[:,:,shijian], data["flow_y"][:,:,shijian])
            Loss.append(loss_RMSE)
            print("Test RMSE: {:02.8f},Test MAE: {:02.8f}".format(loss_RMSE.sqrt(),loss_MAE)) 