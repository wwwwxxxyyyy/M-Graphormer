import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MultiHeadAttention(nn.Module): 
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, in_len):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads 
        self.scale = att_size ** -0.5 
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.Dimension1 = nn.Linear(in_len, 14)
        self.Dimension2 = nn.Linear(in_len, 12)
        self.Dimension3 = nn.Linear(in_len, 8)
        self.Dimension4 = nn.Linear(in_len, 7)
        self.Dimension5 = nn.Linear(in_len, 5)
        self.Dimension6 = nn.Linear(in_len, 1)        
        self.attention_dropout_rate = attention_dropout_rate
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
    def forward(self, q, k, v, attn_bias=None):   #attn_bias (batch_size, in_len, n_node, n_node，num_heads) 
        # layer=1 q=k=v:(batch_size,  seq_len, num_nodes,dialition_size)
        q = q.transpose(1,3)
        k = k.transpose(1,3)
        v = v.transpose(1,3)
        orig_q_size = q.size() # (batch_size, seq_len, num_nodes, hidden_dim)
        d_k = self.att_size 
        d_v = self.att_size
        batch_size = q.size(0)
        seq_len = q.size(1)
        q = self.linear_q(q).view(batch_size, seq_len, -1, self.num_heads, d_k)
        # (batch_size, seq_len, num_nodes, hidden_dim)->(batch_size, seq_len, num_nodes, num_heads * att_size)-> (batch_size, seq_len, num_nodes, num_heads ,d_k)
        k = self.linear_k(k).view(batch_size, seq_len, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, seq_len, -1, self.num_heads, d_v)
        # (batch_size, seq_len, num_nodes, hidden_dim)->(batch_size, seq_len, num_nodes, num_heads * att_size)-> (batch_size, seq_len, num_nodes, num_heads ,d_v)
        q = q.transpose(2, 3) # (batch_sizes, seq_len, num_heads, num_node, d_k)
        v = v.transpose(2, 3) # (batch_sizes, seq_len, num_heads, num_node, d_v)
        k = k.transpose(2, 3).transpose(3, 4) # (batch_sizes, seq_len, num_heads, d_k, num_node)
        q = q * self.scale # (batch_sizes, seq_len, num_heads, num_node, d_k)
        x = torch.matmul(q, k)# (batch_sizes, seq_len, num_heads, num_node, num_node)
        if seq_len==14:
            attn_bias = (self.Dimension1(attn_bias.transpose(1,4))).transpose(1,4) #attn_bias (batch_size, seq_len, n_node, n_node，num_heads) 
        elif seq_len==12:
            attn_bias = (self.Dimension2(attn_bias.transpose(1,4))).transpose(1,4)
        elif seq_len==8:   
            attn_bias = (self.Dimension3(attn_bias.transpose(1,4))).transpose(1,4)
        elif seq_len==7:
            attn_bias = (self.Dimension4(attn_bias.transpose(1,4))).transpose(1,4)
        elif seq_len==5:
            attn_bias = (self.Dimension5(attn_bias.transpose(1,4))).transpose(1,4)
        elif seq_len==1:
            attn_bias = (self.Dimension6(attn_bias.transpose(1,4))).transpose(1,4)
        
        if attn_bias is not None:
            x = x + attn_bias.transpose(3,4).transpose(2,3) # (batch_sizes, seq_len, num_heads, num_node, num_node)

        x = torch.softmax(x, dim=3)
        x = x.matmul(v)  # (batch_sizes, seq_len, num_heads, num_node, d_v)
        x = x.transpose(2, 3).contiguous()  # (batch_sizes, seq_len, num_heads, num_node, d_v) -> (batch_sizes, seq_len, num_node, num_heads, d_v)
        x = x.view(batch_size, seq_len, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        x = F.dropout(x, self.attention_dropout_rate, training=self.training)
#         
        assert x.size() == orig_q_size
        return x  #(batch_sizes, seq_len, num_node, num_heads * att_size)  num_heads * att_size = hidden_size 
class EncoderLayer(nn.Module):  
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,in_len):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads,in_len)
    def forward(self, x, attn_bias=None):
        y = self.self_attention(x, x, x, attn_bias) 
        return y.transpose(1,3)
class TCN_Graphoremer(nn.Module):
    def __init__(self, num_node, dropout, in_dim, out_len, num_heads ,attention_dropout_rate, ffn_dim, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers, in_len):
        super(TCN_Graphoremer, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList() 
        self.graphormer = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,out_channels=residual_channels,kernel_size=(1,1))
        receptive_field = 1 
        self.supports_len = 2 
        for b in range(blocks): 
            additional_scope = 1
            new_dilation = 1 
            for i in range(layers):
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))
                
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                new_dilation *=2 
                receptive_field += additional_scope 
                additional_scope *= 2
                self.graphormer.append(EncoderLayer(dilation_channels ,  ffn_dim, dropout, attention_dropout_rate, num_heads, in_len))
                self.ln.append(nn.LayerNorm([dilation_channels, num_node, (2 ** layers - 1) * blocks + 2 - receptive_field]))
       
        self.end_conv_b1 = nn.Conv1d(in_channels=skip_channels * blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_b2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1,1),
                                    bias=True)
        self.end_conv_c1 = nn.Conv1d(in_channels=skip_channels * blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_c2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1,1),
                                    bias=True)    
        self.receptive_field = receptive_field 

    
    def forward(self, input, attn_bias):
        in_len = input.size(3)  #[batch_size, feature, N, in_len]
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0)) #torch.Size([32, 3, 31, 15])
        else:
            x = input   
        x = self.start_conv(x) #torch.Size([32, 32, 47, 15]) （batch_size, residual_channels, num_nodes, 15）
        skip = 0
        for i in range(self.blocks * self.layers): 
            res = x   #torch.Size([32, 32, 47, 15])
            filter = self.filter_convs[i](x) #torch.Size([32, 32, 47, 14]) 
            #（batch_size, dilation_channels, num_nodes, 15->14->12->8->7->5->1）
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x) #torch.Size([32, 32, 47, 14])  
            #（batch_size, dilation_channels, num_nodes, 15->14->12->8->7->5->1）
            gate = torch.sigmoid(gate)
            x = filter * gate  #torch.Size([32, 32, 47, 14]) 
            s = x  #torch.Size([32, 32, 47, 14]) 
            s = self.skip_convs[i](s)  #(32, 256, 47, 14)
            try:
                skip = torch.cat((s, skip[:, :, :,  -s.size(3):]), dim=1) 
            except:
                skip = s
            x = self.graphormer[i](x, attn_bias) #torch.Size([32, 32, 47, 14]) （batch_size, dilation_channels, num_nodes, 15->14->12->8->7->5->1）
            try:
                dense = dense[:, :, :, -x.size(3):]
            except:
                dense = 0
            dense = res[:, :, :, -x.size(3):] + dense    
            gate = torch.sigmoid(x)  
            x = x * gate + dense * (1 - gate) 
            x = self.ln[i](x)  
        beta = F.relu(skip)
        beta = F.relu(self.end_conv_b1(beta))
        beta = torch.sigmoid(self.end_conv_b2(beta))
        contact = F.relu(skip)
        contact = F.relu(self.end_conv_c1(contact))
        contact = torch.sigmoid(self.end_conv_c2(contact))#(batch_size,out_len,num_node,1)
        return beta,contact
class SIRcell(nn.Module):
    def __init__(self):
        super(SIRcell, self).__init__()

    def forward(self, param_b: torch.Tensor,contact: torch.Tensor, mob: torch.Tensor, SIR: torch.Tensor, sps: torch.Tensor):
        #beta：torch.Size([32, 31，1])，gamma:torch.Size([32, 31, 1]) , Virus_death: torch.Size([ 32, 31, 1])
        #mob[batch_size, 31, 31]
        #contact[batch_size,31,1]
        #sps[batch_size,31,31]
        nature_birth=np.load('birth_china.npy')
        nature_birth=torch.from_numpy(nature_birth).float().to('cuda:0')/365000
        nature_death=np.load('death_china.npy')
        nature_death=torch.from_numpy(nature_death).float().to('cuda:0')/365000

        batch_size = SIR.shape[0]
        num_node = SIR.shape[-2] ##SIR = tensor (32,31,3)
        mob_clone = mob.clone()
        beta = param_b.clone() 

        tensor = torch.ones(batch_size, num_node, num_node)
        mask = torch.eye(tensor.size(1)).bool().unsqueeze(0)
        mask = mask.expand(tensor.size(0), -1, -1)
        tensor.masked_fill_(mask, 0)
        tensor =tensor.to('cuda:0')
        mob_clone = mob_clone * tensor
        # c_out = mob_clone.transpose(1,2)
        sps_clone = sps.clone() * tensor
        F_nm = mob_clone
        F = F_nm.sum(1).unsqueeze(2).expand(batch_size,num_node,num_node) 
        P = F_nm / F

        fai = F_nm.sum(2).sum(1).unsqueeze(1).expand(batch_size,1)  #(batch_size,1)
        nature_birth = nature_birth.expand(batch_size,-1,-1) #(32,31,1)
        nature_death = nature_death.expand(batch_size,-1,-1) #(32,31,1)
        param_g =  0.125 
        Virus_death = 0.00008  #8/100000 #https:// 
        S = SIR[..., [0]]  #torch.Size([32, 31, 1])
        I = SIR[..., [1]]  #torch.Size([32, 31, 1])
        R = SIR[..., [2]]  #torch.Size([32, 31, 1])
        I_sum = SIR[..., [3]]
        pop = S + I + R  #[32,31,1]
        # pop_expand = (S + I + R).expand(-1, num_node, num_node)  #[32,31,31] c_in
        # pop_expand_transpose = pop_expand.transpose(1,2) # c_out

        tau = pop.sum(1) #(batch_size , 1)
        m = (fai / tau).unsqueeze(2).expand(batch_size,num_node,1) #(32,31,1)
        
        I_new_confirm_phy = (S/pop) * beta * contact * I
        
        R_t = R +  param_g * I -  nature_death * R - m * R + ( m.expand(-1,num_node,num_node) * (R.expand(-1,num_node,num_node).transpose(1,2)) * P).sum(2).unsqueeze(2)
        

        I_t = I + I_new_confirm_phy - nature_death * I - param_g * I - Virus_death * I - m * I + ( m.expand(-1,num_node,num_node) * (I.expand(-1,num_node,num_node).transpose(1,2)) * P).sum(2).unsqueeze(2)
        
        S_t = S - I_new_confirm_phy - nature_death * S + nature_birth * pop - m * S + ( m.expand(-1,num_node,num_node) * (S.expand(-1,num_node,num_node).transpose(1,2)) * P).sum(2).unsqueeze(2)

        I_sum_t = I_sum + I_new_confirm_phy

        R0 = (beta * contact) / (nature_death + param_g + Virus_death + m)

        # arrive_time = sps_clone /   (beta * contact - nature_death - param_g - Virus_death - m).expand(-1,num_node,num_node)
        W_arrive = beta * contact - nature_death - param_g - Virus_death #(batch_size,31,1)
        X_arrive = sps_clone.unsqueeze(3) #(batch_size,31,31,1)
        m_arrive = m.clone() #(batch_size,31,1)
        Pd_arrive = P.clone().unsqueeze(3) #(batch_size,31,31,1)
        j0_arrive = I.clone() #(batch_size,31,1)
        N_arrive = pop.clone() #(batch_size,31,1)

        Ht_SIR = torch.cat((R0,I_new_confirm_phy,S_t, I_t, R_t,I_sum_t), dim=-1) #(batch_size, 31, 4)
        arrive_time_para1 = torch.cat((W_arrive,m_arrive,j0_arrive,N_arrive), dim=-1) #(batch_size, 31,4)
        arrive_time_para2 = torch.cat((X_arrive,Pd_arrive), dim=-1) #(batch_size, 31, 31, 4)
        return Ht_SIR, arrive_time_para1, arrive_time_para2

def spd_funtion(spatial_pos:torch.Tensor,k,b):
    #spatial_pos (batch_size,in_len,num_node,num_node,1)
    spatial_pos = - k * spatial_pos - b  # monotonic decrease
    return spatial_pos
    
class spatio(nn.Module):
    def __init__(self, num_node, in_dim, blocks, layers, in_len, out_len, dropout, num_heads=10 ,attention_dropout_rate=0.5, ffn_dim=256, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2):
        super(spatio, self).__init__()
        self.TCN_Graphoremer = TCN_Graphoremer(num_node, dropout, in_dim, out_len, num_heads ,attention_dropout_rate, ffn_dim, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, blocks, layers, in_len)
        self.SIRcell = SIRcell()
        self.out_dim = out_len
        self.num_heads = num_heads
        self.num_node = num_node
        
        self.k = nn.Parameter(torch.empty(1),requires_grad=True)
        self.b = nn.Parameter(torch.empty(1),requires_grad=True)
        nn.init.normal_(self.k, 1, 0.01)
        nn.init.normal_(self.b, 1, 0.01)
        self.inc_init = nn.Parameter(torch.empty(out_len, in_len), requires_grad=True)
        self.inc_init_contact = nn.Parameter(torch.empty(out_len, in_len), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(num_node,1), requires_grad=True)
        nn.init.normal_(self.inc_init, 1, 0.01)
        nn.init.normal_(self.inc_init_contact, 1, 0.01)
        nn.init.normal_(self.bias, 5, 1)
        self.spatial_pos_encoder = nn.Linear(1, num_heads, bias=False)
        self.edge_encoder = nn.Linear(1, num_heads, bias=False)
        

    def forward(self, attn_bias, spatial_pos, x, edge_input, SIR, od, Intra_city_travel_intensity):
        #x(batch_size, in_len, num_node, n_features)
        incidence = torch.softmax(self.inc_init, dim=1) #(in_len,out_len)
        inc_init_contact = torch.softmax(self.inc_init_contact, dim=1)

        mob = torch.einsum('kl,blnm->bknm', incidence, od).squeeze(-1) #(batch_size,out_len,num_node,num_node)
       
        graph_attn_bias = attn_bias.clone() #(n_graph, in_len, n_node, n_node)                                                                       
        graph_attn_bias = graph_attn_bias.unsqueeze(4).repeat(1,1,1,1,self.num_heads)  #(n_graph, in_len, n_node, n_node，num_heads) 
        sps = spatial_pos.clone().squeeze(4) #(batch_size, in_len, num_node, num_node)
        # spd enconding
        k = F.relu(self.k)
        b = F.relu(self.b)
        spatial_pos = spd_funtion(spatial_pos,k,b)
        spatial_pos = torch.sigmoid(spatial_pos)
        
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)  #  (batch_size, in_len, n_node, n_node, num_heads)
#         spatial_pos_bias = spatial_pos.repeat(1,1,1,1,self.num_heads)
        graph_attn_bias= graph_attn_bias + spatial_pos_bias 
        # edge encoding
        edge_input = self.edge_encoder(edge_input) #（batch_size, in_len , n_node, n_node, 1）->（batch_size, in_len, n_node, n_node, num_heads）
#         edge_input = edge_input.repeat(1,1,1,1,self.num_heads)
        graph_attn_bias = graph_attn_bias + edge_input 

        beta, contact_factor= self.TCN_Graphoremer(x.transpose(1,3), graph_attn_bias)  #(32,7,1)
        population_density = np.load('population_density_china.npy')
        population_density=torch.from_numpy(population_density).float().to('cuda:0')

       
        
        Intra_city_travel_intensity_pre = torch.einsum('kl,blnm->bknm', inc_init_contact, Intra_city_travel_intensity)

        batch_size = Intra_city_travel_intensity_pre.shape[0]
        out_len = Intra_city_travel_intensity_pre.shape[1]
        num_node = Intra_city_travel_intensity_pre.shape[2]
        population_density = population_density.reshape(num_node,1).expand(batch_size,out_len,num_node,1)
        contact =  torch.log(torch.pow(Intra_city_travel_intensity_pre * population_density,contact_factor+0.5)) + self.bias
        outputs_new_daily_confirm=[]
        outputs_Cumulative_confirm=[]
        outputs_S=[]
        outputs_I=[]
        outputs_R=[]
        outputs_R0=[]
        outputs_W_arrive = [] #(batch_size,31,1)
        outputs_X_arrive = [] #(batch_size,31,31,1)
        outputs_m_arrive = [] #(batch_size,31,1)
        outputs_Pd_arrive = [] #(batch_size,31,31,1)
        outputs_j0_arrive = [] #(batch_size,31,1)
        outputs_N_arrive = [] #(batch_size,31,1)
        SIR = SIR[:, -1, ...]
        for i in range(self.out_dim):
            NSIR, arrive_time_para1, arrive_time_para2 = self.SIRcell(beta[:,i,...],contact[:,i,...],mob[:,i,...],SIR,sps[:,i,...])
            SIR = NSIR[..., 2:]  
            outputs_R0.append(NSIR[..., [0]])
            outputs_new_daily_confirm.append(NSIR[..., [1]])
            outputs_S.append(NSIR[..., [2]])
            outputs_I.append(NSIR[..., [3]])
            outputs_R.append(NSIR[..., [4]])
            outputs_Cumulative_confirm.append(NSIR[..., [5]])

            outputs_W_arrive.append(arrive_time_para1[..., [0]])
            outputs_m_arrive.append(arrive_time_para1[..., [1]])
            outputs_j0_arrive.append(arrive_time_para1[..., [2]])
            outputs_N_arrive.append(arrive_time_para1[..., [3]])

            outputs_X_arrive.append(arrive_time_para2[..., [0]])
            outputs_Pd_arrive.append(arrive_time_para2[..., [1]])

        outputs_W_arrive_pre = torch.stack(outputs_W_arrive, dim=1)
        outputs_m_arrive_pre = torch.stack(outputs_m_arrive, dim=1)
        outputs_j0_arrive_pre = torch.stack(outputs_j0_arrive, dim=1)
        outputs_N_arrive_pre = torch.stack(outputs_N_arrive, dim=1)

        outputs_X_arrive_pre = torch.stack(outputs_X_arrive, dim=1)
        outputs_Pd_arrive_pre = torch.stack(outputs_Pd_arrive, dim=1)



        outputs_R0_pre = torch.stack(outputs_R0, dim=1)
        outputs_new_daily_confirm_pre = torch.stack(outputs_new_daily_confirm, dim=1)
        outputs_S_pre = torch.stack(outputs_S, dim=1)
        outputs_I_pre = torch.stack(outputs_I, dim=1)
        outputs_R_pre = torch.stack(outputs_R, dim=1)
        outputs_Cumulative_confirm_pre = torch.stack(outputs_Cumulative_confirm, dim=1)
        return outputs_new_daily_confirm_pre,outputs_S_pre,outputs_I_pre, outputs_R_pre, outputs_Cumulative_confirm_pre, beta, contact,outputs_R0_pre, outputs_W_arrive_pre,outputs_m_arrive_pre,outputs_j0_arrive_pre,outputs_N_arrive_pre,outputs_X_arrive_pre,outputs_Pd_arrive_pre