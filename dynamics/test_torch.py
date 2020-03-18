import torch 

device = torch.device("cuda:0") # Uncomment this to run on GPU

a = torch.randn(5,5, device=device)

Q, _ = torch.randn(5, 5, device=device).qr()

print ("a = ",a)

print ("Q = ",Q)

_ , Sa, _ = torch.svd(a, False, False)

_, SQ, _ = torch.svd(Q, False, False)

print ("Sa = ",Sa)

print ("SQ = ",SQ)

# >>> u, s, v = torch.svd(a)
# >>> u
# tensor([[ 0.4027,  0.0287,  0.5434],
# 		[-0.1946,  0.8833,  0.3679],
# 		[ 0.4296, -0.2890,  0.5261],
# 		[ 0.6604,  0.2717, -0.2618],
# 		[ 0.4234,  0.2481, -0.4733]])
# >>> s
# tensor([2.3289, 2.0315, 0.7806])
# >>> v
# tensor([[-0.0199,  0.8766,  0.4809],
# 		[-0.5080,  0.4054, -0.7600],
# 		[ 0.8611,  0.2594, -0.4373]])
# >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
# tensor(8.6531e-07)
# >>> a_big = torch.randn(7, 5, 3)
# >>> u, s, v = torch.svd(a_big)
# >>> torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))
