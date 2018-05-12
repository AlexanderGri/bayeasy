# Bayeasy

Lib allows to use Bayesian Neural Network incredibly easy. You can just use your favorite model/layers and by one transform them to bayesian layers. Enjoy!


## Example of use

```python
import torch
from torch.autograd import Variable
from torch.nn import Linear, Sequential
from torch.nn.functional import mse_loss
from bayeasy import bayesify, get_var_cost

x = Variable(torch.randn(2, 3))
y = Variable(torch.randn(2, 5))

model = Sequential(Linear(3, 5), Linear(5, 5))
bayesify(model[0], n_samples=100)

y_pred = model(x)
loss = mse_loss(y, y_pred) + get_var_cost(model)
loss.backward()
```

After forwarding some input you can estimate uncertainity:

```
print(model[0]._uncertainity)
``` 

For now there is only fully factorized gaussian distribution.