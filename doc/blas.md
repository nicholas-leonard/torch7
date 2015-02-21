<a name="torch.basicoperations.dok"></a>
## Basic operations ##

In this section, we explain basic mathematical operations for Tensors.

<a name="torch.add"></a>
### [res] torch.add([res,] tensor, value) ###
<a name="torch.add"></a>

Add the given value to all elements in the tensor.

`y=torch.add(x,value)` returns a new tensor.

`x:add(value)` add `value` to all elements in place.

<a name="torch.add"></a>
### [res] torch.add([res,] tensor1, tensor2) ###
<a name="torch.add"></a>

Add `tensor1` to `tensor2` and put result into `res`. The number
of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:add(y)
> = x

 5  5
 5  5
[torch.Tensor of dimension 2x2]
```

`y=torch.add(a,b)` returns a new tensor.

`torch.add(y,a,b)` puts `a+b` in `y`.

`a:add(b)` accumulates all elements of `b` into `a`.

`y:add(a,b)` puts `a+b` in `y`.

<a name="torch.add"></a>
### [res] torch.add([res,] tensor1, value, tensor2) ###
<a name="torch.add"></a>

Multiply elements of `tensor2` by the scalar `value` and add it to
`tensor1`. The number of elements must match, but sizes do not
matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:add(2, y)
> = x

 8  8
 8  8
[torch.Tensor of dimension 2x2]
```

`x:add(value,y)` multiply-accumulates values of `y` into `x`.

`z:add(x,value,y)` puts the result of `x + value*y` in `z`.

`torch.add(x,value,y)` returns a new tensor `x + value*y`.

`torch.add(z,x,value,y)` puts the result of `x + value*y` in `z`.

<a name="torch.mul"></a>
### [res] torch.mul([res,] tensor1, value) ###
<a name="torch.mul"></a>

Multiply all elements in the tensor by the given `value`.

`z=torch.mul(x,2)` will return a new tensor with the result of `x*2`.

`torch.mul(z,x,2)` will put the result of `x*2` in `z`.

`x:mul(2)` will multiply all elements of `x` with `2` in-place.

`z:mul(x,2)` will put the result of `x*2` in `z`.

<a name="torch.clamp"></a>
### [res] torch.clamp([res,] tensor1, min_value, max_value) ###
<a name="torch.mul"></a>

Clamp all elements in the tensor into the range `[min_value, max_value]`.  ie:

```
y_i = x_i, if x_i >= min_value or x_i <= max_value
    = min_value, if x_i < min_value
    = max_value, if x_i > max_value
```

`z=torch.clamp(x,0,1)` will return a new tensor with the result of `x` bounded between `0` and `1`.

`torch.clamp(z,x,0,1)` will put the result in `z`.

`x:clamp(0,1)` will perform the clamp operation in place (putting the result in `x`).

`z:clamp(x,0,1)` will put the result in `z`.

<a name="torch.cmul"></a>
### [res] torch.cmul([res,] tensor1, tensor2) ###
<a name="torch.cmul"></a>

Element-wise multiplication of `tensor1` by `tensor2`. The number
of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:cmul(y)
> = x

 6  6
 6  6
[torch.Tensor of dimension 2x2]
```

`z=torch.cmul(x,y)` returns a new tensor.

`torch.cmul(z,x,y)` puts the result in `z`.

`y:cmul(x)` multiplies all elements of `y` with corresponding elements of `x`.

`z:cmul(x,y)` puts the result in `z`.

<a name="torch.cpow"></a>
### [res] torch.cpow([res,] tensor1, tensor2) ###
<a name="torch.cpow"></a>

Element-wise power operation, taking the elements of `tensor1` to the powers
given by elements of `tensor2`. The number of elements must match,
but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> x:cpow(y)
> = x

 8  8
 8  8
[torch.Tensor of dimension 2x2]
```

`z=torch.cpow(x,y)` returns a new tensor.

`torch.cpow(z,x,y)` puts the result in `z`.

`y:cpow(x)` takes all elements of `y` to the powers given by the
corresponding elements of `x`.

`z:cpow(x,y)` puts the result in `z`.

<a name="torch.addcmul"></a>
### [res] torch.addcmul([res,] x [,value], tensor1, tensor2) ###
<a name="torch.addcmul"></a>

Performs the element-wise multiplication of `tensor1` by `tensor2`,
multiply the result by the scalar `value` (1 if not present) and add it
to `x`. The number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> z = torch.Tensor(2,2):fill(5)
> x:addcmul(2, y, z)
> = x

 32  32
 32  32
[torch.Tensor of dimension 2x2]
```

`z:addcmul(value,x,y)` accumulates the result in `z`.

`torch.addcmul(z,value,x,y)` returns a new tensor with the result.

`torch.addcmul(z,z,value,x,y)` puts the result in `z`.

<a name="torch.div"></a>
### [res] torch.div([res,] tensor, value) ###
<a name="torch.div"></a>

Divide all elements in the tensor by the given `value`.

`z=torch.div(x,2)` will return a new tensor with the result of `x/2`.

`torch.div(z,x,2)` will put the result of `x/2` in `z`.

`x:div(2)` will divide all elements of `x` with `2` in-place.

`z:div(x,2)` with put the result of `x/2` in `z`.

<a name="torch.cdiv"></a>
### [res] torch.cdiv([res,] tensor1, tensor2) ###
<a name="torch.cdiv"></a>

Performs the element-wise division of `tensor1` by `tensor2`.  The
number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(1)
> y = torch.Tensor(4)        
> for i=1,4 do y[i] = i end
> x:cdiv(y)
> = x

 1.0000  0.3333
 0.5000  0.2500
[torch.Tensor of dimension 2x2]
```

`z=torch.cdiv(x,y)` returns a new tensor.

`torch.cdiv(z,x,y)` puts the result in `z`.

`y:cdiv(x)` divides all elements of `y` with corresponding elements of `x`.

`z:cdiv(x,y)` puts the result in `z`.

<a name="torch.addcdiv"></a>
### [res] torch.addcdiv([res,] x [,value], tensor1, tensor2) ###
<a name="torch.addcdiv"></a>

Performs the element-wise division of `tensor1` by `tensor1`, 
multiply the result by the scalar `value` and add it to `x`. 
The number of elements must match, but sizes do not matter.

```
> x = torch.Tensor(2,2):fill(1)
> y = torch.Tensor(4)
> z = torch.Tensor(2,2):fill(5)
> for i=1,4 do y[i] = i end
> x:addcdiv(2, y, z)
> = x

 1.4000  2.2000
 1.8000  2.6000
[torch.Tensor of dimension 2x2]
```

`z:addcdiv(value,x,y)` accumulates the result in `z`.

`torch.addcdiv(z,value,x,y)` returns a new tensor with the result.

`torch.addcdiv(z,z,value,x,y)` puts the result in `z`.

<a name="torch.dot"></a>
### [number] torch.dot(tensor1,tensor2) ###
<a name="torch.dot"></a>

Performs the dot product between `tensor` and self. The number of
elements must match: both tensors are seen as a 1D vector.

```
> x = torch.Tensor(2,2):fill(2)
> y = torch.Tensor(4):fill(3)
> = x:dot(y)
24
```

`torch.dot(x,y)` returns dot product of `x` and `y`.
`x:dot(y)` returns dot product of `x` and `y`.

<a name="torch.addmv"></a>
### [res] torch.addmv([res,] [beta,] [v1,] vec1, [v2,] mat, vec2) ###
<a name="torch.addmv"></a>

Performs a matrix-vector multiplication between `mat` (2D tensor)
and `vec` (1D tensor) and add it to vec1. In other words,

```
res = beta * res + v1 * vec1 + v2 * mat*vec2
```

Sizes must respect the matrix-multiplication operation: if `mat` is
a `n x m` matrix, `vec2` must be vector of size `m` and `vec1` must
be a vector of size `n`.

```
> x = torch.Tensor(3):fill(0)
> M = torch.Tensor(3,2):fill(3)
> y = torch.Tensor(2):fill(2)
> x:addmv(M, y)
> = x

 12
 12
 12
[torch.Tensor of dimension 3]
```

`torch.addmv(x,y,z)` returns a new tensor with the result.

`torch.addmv(r,x,y,z)` puts the result in `r`.

`x:addmv(y,z)` accumulates `y*z` into `x`.

`r:addmv(x,y,z)` puts the result of `x+y*z` into `r`.

Optional values `v1` and `v2` are scalars that multiply 
`vec1` and `mat*vec2` respectively.

Optional value `beta` is  a scalar that scales the result tensor, before accumulating the result into the tensor. Defaults to 1.0

<a name="torch.addr"></a>
### [res] torch.addr([res,] [v1,] mat, [v2,] vec1, vec2) ###
<a name="torch.addr"></a>

Performs the outer-product between `vec1` (1D tensor) and `vec2` (1D tensor).
In other words,

```
res_ij = v1 * mat_ij + v2 * vec1_i * vec2_j
```

If `vec1` is a vector of size `n` and `vec2` is a vector of size `m`, 
then mat must be a matrix of size `n x m`.

```
> x = torch.Tensor(3)        
> y = torch.Tensor(2)
> for i=1,3 do x[i] = i end
> for i=1,2 do y[i] = i end
> M = torch.Tensor(3, 2):zero()
> M:addr(x, y)
> = M

 1  2
 2  4
 3  6
[torch.Tensor of dimension 3x2]
```

`torch.addr(M,x,y)` returns the result in a new tensor.

`torch.addr(r,M,x,y)` puts the result in `r`.

`M:addr(x,y)` puts the result in `M`.

`r:addr(M,x,y)` puts the result in `r`.

Optional values `v1` and `v2` are scalars that multiply 
`M` and `vec1 [out] vec2` respectively.


<a name="torch.addmm"></a>
### [res] torch.addmm([res,] [beta,] [v1,] M [v2,] mat1, mat2) ###
<a name="torch.addmm"></a>

Performs a matrix-matrix multiplication between `mat1` (2D tensor)
and `mat2` (2D tensor). In other words,

```
res = res * beta + v1 * M + v2 * mat1*mat2
```

If `mat1` is a `n x m` matrix, `mat2` a `m x p` matrix, 
`M` must be a `n x p` matrix.

`torch.addmm(M,mat1,mat2)` returns the result in a new tensor.

`torch.addmm(r,M,mat1,mat2)` puts the result in `r`.

`M:addmm(mat1,mat2)` puts the result in `M`.

`r:addmm(M,mat1,mat2)` puts the result in `r`.

Optional values `v1` and `v2` are scalars that multiply 
`M` and `mat1 * mat2` respectively.

Optional value `beta` is  a scalar that scales the result tensor, before accumulating the result into the tensor. Defaults to 1.0

<a name="torch.addbmm"></a>
### [res] torch.addbmm([res,] [v1,] M [v2,] mat1, mat2) ###
<a name="torch.addbmm"></a>

Batch matrix matrix product of matrices stored in `batch1` and `batch2`, 
with a reduced add step (all matrix multiplications get accumulated in a
single place).
`batch1` and `batch2` must be 3D tensors each containing the same number
of matrices. If `batch1` is a `b x n x m` tensor, `batch2` a `b x m x p`
tensor, res will be a `n x p` tensor.

`torch.addbmm(M,x,y)` puts the result in a new tensor.

`M:addbmm(x,y)` puts the result in `M`, resizing `M` if necessary.

`M:addbmm(beta,M2,alpha,x,y)` puts the result in `M`, resizing `M` if necessary.

<a name="torch.baddbmm"></a>
### [res] torch.baddbmm([res,] [v1,] M [v2,] mat1, mat2) ###
<a name="torch.baddbmm"></a>

Batch matrix matrix product of matrices stored in `batch1` and `batch2`,
with batch add.
`batch1` and `batch2` must be 3D tensors each containing the same number
of matrices. If `batch1` is a `b x n x m` tensor, `batch2` a `b x m x p`
tensor, res will be a `b x n x p` tensor.

`torch.baddbmm(M,x,y)` puts the result in a new tensor.

`M:baddbmm(x,y)` puts the result in `M`, resizing `M` if necessary.

`M:baddbmm(beta,M2,alpha,x,y)` puts the result in `M`, resizing `M` if necessary.

<a name="torch.mv"></a>
### [res] torch.mv([res,] mat, vec) ###
<a name="torch.mv"></a>

Matrix vector product of `mat` and `vec`. Sizes must respect 
the matrix-multiplication operation: if `mat` is a `n x m` matrix, 
`vec` must be vector of size `m` and res must be a vector of size `n`.

`torch.mv(x,y)` puts the result in a new tensor.

`torch.mv(M,x,y)` puts the result in `M`.

`M:mv(x,y)` puts the result in `M`.

<a name="torch.mm"></a>
### [res] torch.mm([res,] mat1, mat2) ###
<a name="torch.mm"></a>

Matrix matrix product of `mat1` and `mat2`. If `mat1` is a 
`n x m` matrix, `mat2` a `m x p` matrix, res must be a 
`n x p` matrix.


`torch.mm(x,y)` puts the result in a new tensor.

`torch.mm(M,x,y)` puts the result in `M`.

`M:mm(x,y)` puts the result in `M`.

<a name="torch.bmm"></a>
### [res] torch.bmm([res,] batch1, batch2) ###
<a name="torch.bmm"></a>

Batch matrix matrix product of matrices stored in `batch1` and `batch2`.
`batch1` and `batch2` must be 3D tensors each containing the same number
of matrices. If `batch1` is a `b x n x m` tensor, `batch2` a `b x m x p`
tensor, res will be a `b x n x p` tensor.


`torch.bmm(x,y)` puts the result in a new tensor.

`torch.bmm(M,x,y)` puts the result in `M`, resizing `M` if necessary.

`M:bmm(x,y)` puts the result in `M`, resizing `M` if necessary.

<a name="torch.ger"></a>
### [res] torch.ger([res,] vec1, vec2) ###
<a name="torch.ger"></a>

Outer product of `vec1` and `vec2`. If `vec1` is a vector of 
size `n` and `vec2` is a vector of size `m`, then res must 
be a matrix of size `n x m`.


`torch.ger(x,y)` puts the result in a new tensor.

`torch.ger(M,x,y)` puts the result in `M`.

`M:ger(x,y)` puts the result in `M`.

