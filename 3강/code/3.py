import torch

print(torch.cuda.is_available())

tc = torch.tensor([[1., -1.], [1., -1.]])
print(tc)

print(torch.tensor([[1, 2], [3, 4]]))
print(torch.tensor([[1, 2], [3, 4]], device='cuda:0'))
print(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64))

v = torch.tensor([1, 2, 3])
w = torch.tensor([3, 4, 6])
print(w - v)

temp = torch.tensor([[1, 2], [3, 4]])
print(temp.shape)
print('-----------')
print(temp.size())
print('-----------')
print(temp.view(4, 1)) # 2*2 를 4*1 로 변형
print('-----------')
print(temp.view(-1)) # 2*2 를 1차원 벡터로
print('-----------')
print(temp.view(1, -1)) # (1, -1) 은 (1, ?) 와 같은 의미, 원소개수 2*2=4 를 유지한채 (1,?) 의 형태를 만든다는것. -> (1,4)
print('-----------')
print(temp.view(-1, 1)) # (-1, 1) 은 (?, 1) 와 같은 의미, 원소개수 2*2=4 를 유지한채 (?,1) 의 형태를 만든다는것. -> (4,1)
