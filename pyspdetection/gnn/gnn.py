
import torch
from torch_geometric.nn import DenseGINConv
from torch.nn import Sequential, Linear, ReLU, LSTM

device = torch.device("cpu")

class SISNet(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(SISNet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		nn = Sequential(Linear(in_channels, 16), ReLU(), Linear(16, 64), ReLU(), Linear(64, out_channels))
		self.conv = DenseGINConv(nn, self.out_channels)

		
	def forward(self, x, adj):
		x = self.conv(x, adj)
		return torch.sigmoid(x)


class LotkaNet(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(LotkaNet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.linear1 = Linear(in_channels, 32)

		nn = Sequential(Linear(32, 64), ReLU(), Linear(64, 128), ReLU(), Linear(128, 64))
		self.conv1 = DenseGINConv(nn, 64)

		nn = Sequential(Linear(32, 64), ReLU(), Linear(64, 128), ReLU(), Linear(128, 64))
		self.conv2 = DenseGINConv(nn, 64)

		self.linear2 = Linear(64*2, self.out_channels)
		
	def forward(self, x, adj):
		x = self.linear1(x)
		x1 = self.conv1(x, adj)
		x2 = self.conv2(x, torch.transpose(adj, 1, 2))
		x3 = torch.cat([x1, x2], dim=2) 
		return self.linear2(x3)



class ThetaNet(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(ThetaNet, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.linear1 = Linear(in_channels, 32)

		nn = Sequential(Linear(32, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 32))
		self.conv1 = DenseGINConv(nn, 32)

		nn = Sequential(Linear(32, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 32))
		self.conv2 = DenseGINConv(nn, 32)

		self.linear2 = Linear(64, self.out_channels)
		
	def forward(self, x, adj):
		x = self.linear1(x)
		x1 = self.conv1(x, adj)
		x2 = self.conv2(x, torch.transpose(adj, 1, 2))
		x3 = torch.cat([x1, x2], dim=2) 
		return torch.sigmoid(self.linear2(x3))
