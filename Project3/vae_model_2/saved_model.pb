��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:*
dtype0
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	�z	*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:	*
dtype0
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	�z	*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
deconv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv1/kernel
y
"deconv1/kernel/Read/ReadVariableOpReadVariableOpdeconv1/kernel*&
_output_shapes
:*
dtype0
p
deconv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv1/bias
i
 deconv1/bias/Read/ReadVariableOpReadVariableOpdeconv1/bias*
_output_shapes
:*
dtype0
�
deconv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv2/kernel
y
"deconv2/kernel/Read/ReadVariableOpReadVariableOpdeconv2/kernel*&
_output_shapes
:*
dtype0
p
deconv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedeconv2/bias
i
 deconv2/bias/Read/ReadVariableOpReadVariableOpdeconv2/bias*
_output_shapes
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/m
�
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2/kernel/m
�
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv2/bias/m
s
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes
:*
dtype0
�
Adam/z_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*(
shared_nameAdam/z_log_var/kernel/m
�
+Adam/z_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/m*
_output_shapes
:	�z	*
dtype0
�
Adam/z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/z_log_var/bias/m
{
)Adam/z_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/m*
_output_shapes
:	*
dtype0
�
Adam/z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*%
shared_nameAdam/z_mean/kernel/m
~
(Adam/z_mean/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/m*
_output_shapes
:	�z	*
dtype0
|
Adam/z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/z_mean/bias/m
u
&Adam/z_mean/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/m*
_output_shapes
:	*
dtype0
�
Adam/deconv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/deconv1/kernel/m
�
)Adam/deconv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/deconv1/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/deconv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/deconv1/bias/m
w
'Adam/deconv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/deconv1/bias/m*
_output_shapes
:*
dtype0
�
Adam/deconv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/deconv2/kernel/m
�
)Adam/deconv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/deconv2/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/deconv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/deconv2/bias/m
w
'Adam/deconv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/deconv2/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
��*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv1/kernel/v
�
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/conv2/kernel/v
�
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/conv2/bias/v
s
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes
:*
dtype0
�
Adam/z_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*(
shared_nameAdam/z_log_var/kernel/v
�
+Adam/z_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/v*
_output_shapes
:	�z	*
dtype0
�
Adam/z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/z_log_var/bias/v
{
)Adam/z_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/v*
_output_shapes
:	*
dtype0
�
Adam/z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�z	*%
shared_nameAdam/z_mean/kernel/v
~
(Adam/z_mean/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/v*
_output_shapes
:	�z	*
dtype0
|
Adam/z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/z_mean/bias/v
u
&Adam/z_mean/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/v*
_output_shapes
:	*
dtype0
�
Adam/deconv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/deconv1/kernel/v
�
)Adam/deconv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/deconv1/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/deconv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/deconv1/bias/v
w
'Adam/deconv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/deconv1/bias/v*
_output_shapes
:*
dtype0
�
Adam/deconv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/deconv2/kernel/v
�
)Adam/deconv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/deconv2/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/deconv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/deconv2/bias/v
w
'Adam/deconv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/deconv2/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
��*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�_
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�_
value�_B�_ B�_
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
	layer_with_weights-2
	layer-5
layer_with_weights-3
layer-6
 layer-7
!	variables
"trainable_variables
#regularization_losses
$	keras_api
�
%layer-0
&layer-1
'layer_with_weights-0
'layer-2
(layer_with_weights-1
(layer-3
)layer-4
*layer_with_weights-2
*layer-5
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api

O	keras_api

P	keras_api

Q	keras_api

R	keras_api

S	keras_api

T	keras_api

U	keras_api

V	keras_api

W	keras_api

X	keras_api

Y	keras_api

Z	keras_api

[	keras_api

\	keras_api
R
]	variables
^trainable_variables
_regularization_losses
`	keras_api
�
aiter

bbeta_1

cbeta_2
	ddecay
elearning_rate3m�4m�9m�:m�Cm�Dm�Im�Jm�fm�gm�hm�im�jm�km�3v�4v�9v�:v�Cv�Dv�Iv�Jv�fv�gv�hv�iv�jv�kv�
 
f
30
41
92
:3
I4
J5
C6
D7
f8
g9
h10
i11
j12
k13
f
30
41
92
:3
I4
J5
C6
D7
f8
g9
h10
i11
j12
k13
 
�
lnon_trainable_variables
	variables
mmetrics
trainable_variables
regularization_losses
nlayer_metrics

olayers
player_regularization_losses
 
R
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
8
30
41
92
:3
I4
J5
C6
D7
8
30
41
92
:3
I4
J5
C6
D7
 
�
unon_trainable_variables
!	variables
vmetrics
"trainable_variables
#regularization_losses
wlayer_metrics

xlayers
ylayer_regularization_losses
 
R
z	variables
{trainable_variables
|regularization_losses
}	keras_api
j

fkernel
gbias
~	variables
trainable_variables
�regularization_losses
�	keras_api
l

hkernel
ibias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
l

jkernel
kbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
*
f0
g1
h2
i3
j4
k5
*
f0
g1
h2
i3
j4
k5
 
�
�non_trainable_variables
+	variables
�metrics
,trainable_variables
-regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�metrics
/	variables
0trainable_variables
1regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
�
�non_trainable_variables
�metrics
5	variables
6trainable_variables
7regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
�
�non_trainable_variables
�metrics
;	variables
<trainable_variables
=regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�metrics
?	variables
@trainable_variables
Aregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
�
�non_trainable_variables
�metrics
E	variables
Ftrainable_variables
Gregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
�
�non_trainable_variables
�metrics
K	variables
Ltrainable_variables
Mregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
�non_trainable_variables
�metrics
]	variables
^trainable_variables
_regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdeconv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdeconv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdeconv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdeconv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 

�0
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 
 
 
 
�
�non_trainable_variables
�metrics
q	variables
rtrainable_variables
sregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
8
0
1
2
3
4
	5
6
 7
 
 
 
 
�
�non_trainable_variables
�metrics
z	variables
{trainable_variables
|regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses

f0
g1

f0
g1
 
�
�non_trainable_variables
�metrics
~	variables
trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses

h0
i1

h0
i1
 
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses

j0
k1

j0
k1
 
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
 
 
 
*
%0
&1
'2
(3
)4
*5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/z_log_var/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/z_log_var/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/z_mean/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_mean/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/deconv1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/deconv1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/deconv2/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/deconv2/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/z_log_var/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/z_log_var/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/z_mean/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_mean/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/deconv1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/deconv1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/deconv2/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/deconv2/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_encoder_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdeconv1/kerneldeconv1/biasdeconv2/kerneldeconv2/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_162048
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp"deconv1/kernel/Read/ReadVariableOp deconv1/bias/Read/ReadVariableOp"deconv2/kernel/Read/ReadVariableOp deconv2/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp+Adam/z_log_var/kernel/m/Read/ReadVariableOp)Adam/z_log_var/bias/m/Read/ReadVariableOp(Adam/z_mean/kernel/m/Read/ReadVariableOp&Adam/z_mean/bias/m/Read/ReadVariableOp)Adam/deconv1/kernel/m/Read/ReadVariableOp'Adam/deconv1/bias/m/Read/ReadVariableOp)Adam/deconv2/kernel/m/Read/ReadVariableOp'Adam/deconv2/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp+Adam/z_log_var/kernel/v/Read/ReadVariableOp)Adam/z_log_var/bias/v/Read/ReadVariableOp(Adam/z_mean/kernel/v/Read/ReadVariableOp&Adam/z_mean/bias/v/Read/ReadVariableOp)Adam/deconv1/kernel/v/Read/ReadVariableOp'Adam/deconv1/bias/v/Read/ReadVariableOp)Adam/deconv2/kernel/v/Read/ReadVariableOp'Adam/deconv2/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_163171
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasz_log_var/kernelz_log_var/biasz_mean/kernelz_mean/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedeconv1/kerneldeconv1/biasdeconv2/kerneldeconv2/biasdense_1/kerneldense_1/biastotalcountAdam/conv1/kernel/mAdam/conv1/bias/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/z_log_var/kernel/mAdam/z_log_var/bias/mAdam/z_mean/kernel/mAdam/z_mean/bias/mAdam/deconv1/kernel/mAdam/deconv1/bias/mAdam/deconv2/kernel/mAdam/deconv2/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv1/kernel/vAdam/conv1/bias/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/z_log_var/kernel/vAdam/z_log_var/bias/vAdam/z_mean/kernel/vAdam/z_mean/bias/vAdam/deconv1/kernel/vAdam/deconv1/bias/vAdam/deconv2/kernel/vAdam/deconv2/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_163328��
�
}
(__inference_deconv2_layer_call_fn_161372

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv2_layer_call_and_return_conditional_losses_1613622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�N
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_162740

inputs4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��deconv1/BiasAdd/ReadVariableOp�'deconv1/conv2d_transpose/ReadVariableOp�deconv2/BiasAdd/ReadVariableOp�'deconv2/conv2d_transpose/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOpX
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_3/Shape�
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack�
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1�
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2x
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/3�
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape�
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_3/Reshapeh
deconv1/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
deconv1/Shape�
deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice/stack�
deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_1�
deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_2�
deconv1/strided_sliceStridedSlicedeconv1/Shape:output:0$deconv1/strided_slice/stack:output:0&deconv1/strided_slice/stack_1:output:0&deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_sliced
deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/1d
deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/2d
deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/3�
deconv1/stackPackdeconv1/strided_slice:output:0deconv1/stack/1:output:0deconv1/stack/2:output:0deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv1/stack�
deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice_1/stack�
deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_1�
deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_2�
deconv1/strided_slice_1StridedSlicedeconv1/stack:output:0&deconv1/strided_slice_1/stack:output:0(deconv1/strided_slice_1/stack_1:output:0(deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_slice_1�
'deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv1/conv2d_transpose/ReadVariableOp�
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
deconv1/conv2d_transpose�
deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv1/BiasAdd/ReadVariableOp�
deconv1/BiasAddBiasAdd!deconv1/conv2d_transpose:output:0&deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
deconv1/BiasAddx
deconv1/ReluReludeconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
deconv1/Reluh
deconv2/ShapeShapedeconv1/Relu:activations:0*
T0*
_output_shapes
:2
deconv2/Shape�
deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice/stack�
deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_1�
deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_2�
deconv2/strided_sliceStridedSlicedeconv2/Shape:output:0$deconv2/strided_slice/stack:output:0&deconv2/strided_slice/stack_1:output:0&deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_sliced
deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/1d
deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/2d
deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/3�
deconv2/stackPackdeconv2/strided_slice:output:0deconv2/stack/1:output:0deconv2/stack/2:output:0deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv2/stack�
deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice_1/stack�
deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_1�
deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_2�
deconv2/strided_slice_1StridedSlicedeconv2/stack:output:0&deconv2/strided_slice_1/stack:output:0(deconv2/strided_slice_1/stack_1:output:0(deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_slice_1�
'deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv2/conv2d_transpose/ReadVariableOp�
deconv2/conv2d_transposeConv2DBackpropInputdeconv2/stack:output:0/deconv2/conv2d_transpose/ReadVariableOp:value:0deconv1/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
deconv2/conv2d_transpose�
deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv2/BiasAdd/ReadVariableOp�
deconv2/BiasAddBiasAdd!deconv2/conv2d_transpose:output:0&deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
deconv2/BiasAddx
deconv2/ReluReludeconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
deconv2/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_3/Const�
flatten_3/ReshapeReshapedeconv2/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_3/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddz
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Sigmoid�
IdentityIdentitydense_1/Sigmoid:y:0^deconv1/BiasAdd/ReadVariableOp(^deconv1/conv2d_transpose/ReadVariableOp^deconv2/BiasAdd/ReadVariableOp(^deconv2/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2@
deconv1/BiasAdd/ReadVariableOpdeconv1/BiasAdd/ReadVariableOp2R
'deconv1/conv2d_transpose/ReadVariableOp'deconv1/conv2d_transpose/ReadVariableOp2@
deconv2/BiasAdd/ReadVariableOpdeconv2/BiasAdd/ReadVariableOp2R
'deconv2/conv2d_transpose/ReadVariableOp'deconv2/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
r
F__inference_add_loss_1_layer_call_and_return_conditional_losses_161711

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162378

inputs7
3encoder_output_conv1_conv2d_readvariableop_resource8
4encoder_output_conv1_biasadd_readvariableop_resource7
3encoder_output_conv2_conv2d_readvariableop_resource8
4encoder_output_conv2_biasadd_readvariableop_resource8
4encoder_output_z_mean_matmul_readvariableop_resource9
5encoder_output_z_mean_biasadd_readvariableop_resource;
7encoder_output_z_log_var_matmul_readvariableop_resource<
8encoder_output_z_log_var_biasadd_readvariableop_resourceC
?decoder_output_deconv1_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv1_biasadd_readvariableop_resourceC
?decoder_output_deconv2_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv2_biasadd_readvariableop_resource9
5decoder_output_dense_1_matmul_readvariableop_resource:
6decoder_output_dense_1_biasadd_readvariableop_resource
identity

identity_1��conv1/BiasAdd/ReadVariableOp�conv1/Conv2D/ReadVariableOp�conv2/BiasAdd/ReadVariableOp�conv2/Conv2D/ReadVariableOp�-decoder_output/deconv1/BiasAdd/ReadVariableOp�6decoder_output/deconv1/conv2d_transpose/ReadVariableOp�-decoder_output/deconv2/BiasAdd/ReadVariableOp�6decoder_output/deconv2/conv2d_transpose/ReadVariableOp�-decoder_output/dense_1/BiasAdd/ReadVariableOp�,decoder_output/dense_1/MatMul/ReadVariableOp�+encoder_output/conv1/BiasAdd/ReadVariableOp�*encoder_output/conv1/Conv2D/ReadVariableOp�+encoder_output/conv2/BiasAdd/ReadVariableOp�*encoder_output/conv2/Conv2D/ReadVariableOp�/encoder_output/z_log_var/BiasAdd/ReadVariableOp�.encoder_output/z_log_var/MatMul/ReadVariableOp�,encoder_output/z_mean/BiasAdd/ReadVariableOp�+encoder_output/z_mean/MatMul/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpv
encoder_output/reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2 
encoder_output/reshape_2/Shape�
,encoder_output/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,encoder_output/reshape_2/strided_slice/stack�
.encoder_output/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.encoder_output/reshape_2/strided_slice/stack_1�
.encoder_output/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.encoder_output/reshape_2/strided_slice/stack_2�
&encoder_output/reshape_2/strided_sliceStridedSlice'encoder_output/reshape_2/Shape:output:05encoder_output/reshape_2/strided_slice/stack:output:07encoder_output/reshape_2/strided_slice/stack_1:output:07encoder_output/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&encoder_output/reshape_2/strided_slice�
(encoder_output/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder_output/reshape_2/Reshape/shape/1�
(encoder_output/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder_output/reshape_2/Reshape/shape/2�
(encoder_output/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(encoder_output/reshape_2/Reshape/shape/3�
&encoder_output/reshape_2/Reshape/shapePack/encoder_output/reshape_2/strided_slice:output:01encoder_output/reshape_2/Reshape/shape/1:output:01encoder_output/reshape_2/Reshape/shape/2:output:01encoder_output/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&encoder_output/reshape_2/Reshape/shape�
 encoder_output/reshape_2/ReshapeReshapeinputs/encoder_output/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2"
 encoder_output/reshape_2/Reshape�
*encoder_output/conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv1/Conv2D/ReadVariableOp�
encoder_output/conv1/Conv2DConv2D)encoder_output/reshape_2/Reshape:output:02encoder_output/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
encoder_output/conv1/Conv2D�
+encoder_output/conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv1/BiasAdd/ReadVariableOp�
encoder_output/conv1/BiasAddBiasAdd$encoder_output/conv1/Conv2D:output:03encoder_output/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
encoder_output/conv1/BiasAdd�
encoder_output/conv1/ReluRelu%encoder_output/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
encoder_output/conv1/Relu�
*encoder_output/conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv2/Conv2D/ReadVariableOp�
encoder_output/conv2/Conv2DConv2D'encoder_output/conv1/Relu:activations:02encoder_output/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
encoder_output/conv2/Conv2D�
+encoder_output/conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv2/BiasAdd/ReadVariableOp�
encoder_output/conv2/BiasAddBiasAdd$encoder_output/conv2/Conv2D:output:03encoder_output/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
encoder_output/conv2/BiasAdd�
encoder_output/conv2/ReluRelu%encoder_output/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
encoder_output/conv2/Relu�
encoder_output/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2 
encoder_output/flatten_2/Const�
 encoder_output/flatten_2/ReshapeReshape'encoder_output/conv2/Relu:activations:0'encoder_output/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2"
 encoder_output/flatten_2/Reshape�
+encoder_output/z_mean/MatMul/ReadVariableOpReadVariableOp4encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02-
+encoder_output/z_mean/MatMul/ReadVariableOp�
encoder_output/z_mean/MatMulMatMul)encoder_output/flatten_2/Reshape:output:03encoder_output/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
encoder_output/z_mean/MatMul�
,encoder_output/z_mean/BiasAdd/ReadVariableOpReadVariableOp5encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02.
,encoder_output/z_mean/BiasAdd/ReadVariableOp�
encoder_output/z_mean/BiasAddBiasAdd&encoder_output/z_mean/MatMul:product:04encoder_output/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
encoder_output/z_mean/BiasAdd�
.encoder_output/z_log_var/MatMul/ReadVariableOpReadVariableOp7encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype020
.encoder_output/z_log_var/MatMul/ReadVariableOp�
encoder_output/z_log_var/MatMulMatMul)encoder_output/flatten_2/Reshape:output:06encoder_output/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2!
encoder_output/z_log_var/MatMul�
/encoder_output/z_log_var/BiasAdd/ReadVariableOpReadVariableOp8encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype021
/encoder_output/z_log_var/BiasAdd/ReadVariableOp�
 encoder_output/z_log_var/BiasAddBiasAdd)encoder_output/z_log_var/MatMul:product:07encoder_output/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2"
 encoder_output/z_log_var/BiasAdd�
encoder_output/z/ShapeShape&encoder_output/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder_output/z/Shape�
$encoder_output/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$encoder_output/z/strided_slice/stack�
&encoder_output/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_1�
&encoder_output/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_2�
encoder_output/z/strided_sliceStridedSliceencoder_output/z/Shape:output:0-encoder_output/z/strided_slice/stack:output:0/encoder_output/z/strided_slice/stack_1:output:0/encoder_output/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
encoder_output/z/strided_slice�
&encoder_output/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2(
&encoder_output/z/random_normal/shape/1�
$encoder_output/z/random_normal/shapePack'encoder_output/z/strided_slice:output:0/encoder_output/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$encoder_output/z/random_normal/shape�
#encoder_output/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder_output/z/random_normal/mean�
%encoder_output/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%encoder_output/z/random_normal/stddev�
3encoder_output/z/random_normal/RandomStandardNormalRandomStandardNormal-encoder_output/z/random_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2���25
3encoder_output/z/random_normal/RandomStandardNormal�
"encoder_output/z/random_normal/mulMul<encoder_output/z/random_normal/RandomStandardNormal:output:0.encoder_output/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2$
"encoder_output/z/random_normal/mul�
encoder_output/z/random_normalAdd&encoder_output/z/random_normal/mul:z:0,encoder_output/z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2 
encoder_output/z/random_normal�
encoder_output/z/ExpExp)encoder_output/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/Exp�
encoder_output/z/mulMulencoder_output/z/Exp:y:0"encoder_output/z/random_normal:z:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/mul�
encoder_output/z/addAddV2&encoder_output/z_mean/BiasAdd:output:0encoder_output/z/mul:z:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/add�
decoder_output/reshape_3/ShapeShapeencoder_output/z/add:z:0*
T0*
_output_shapes
:2 
decoder_output/reshape_3/Shape�
,decoder_output/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/reshape_3/strided_slice/stack�
.decoder_output/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_3/strided_slice/stack_1�
.decoder_output/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_3/strided_slice/stack_2�
&decoder_output/reshape_3/strided_sliceStridedSlice'decoder_output/reshape_3/Shape:output:05decoder_output/reshape_3/strided_slice/stack:output:07decoder_output/reshape_3/strided_slice/stack_1:output:07decoder_output/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/reshape_3/strided_slice�
(decoder_output/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/1�
(decoder_output/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/2�
(decoder_output/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/3�
&decoder_output/reshape_3/Reshape/shapePack/decoder_output/reshape_3/strided_slice:output:01decoder_output/reshape_3/Reshape/shape/1:output:01decoder_output/reshape_3/Reshape/shape/2:output:01decoder_output/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&decoder_output/reshape_3/Reshape/shape�
 decoder_output/reshape_3/ReshapeReshapeencoder_output/z/add:z:0/decoder_output/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2"
 decoder_output/reshape_3/Reshape�
decoder_output/deconv1/ShapeShape)decoder_output/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2
decoder_output/deconv1/Shape�
*decoder_output/deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_output/deconv1/strided_slice/stack�
,decoder_output/deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv1/strided_slice/stack_1�
,decoder_output/deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv1/strided_slice/stack_2�
$decoder_output/deconv1/strided_sliceStridedSlice%decoder_output/deconv1/Shape:output:03decoder_output/deconv1/strided_slice/stack:output:05decoder_output/deconv1/strided_slice/stack_1:output:05decoder_output/deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_output/deconv1/strided_slice�
decoder_output/deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/1�
decoder_output/deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/2�
decoder_output/deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/3�
decoder_output/deconv1/stackPack-decoder_output/deconv1/strided_slice:output:0'decoder_output/deconv1/stack/1:output:0'decoder_output/deconv1/stack/2:output:0'decoder_output/deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/deconv1/stack�
,decoder_output/deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/deconv1/strided_slice_1/stack�
.decoder_output/deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv1/strided_slice_1/stack_1�
.decoder_output/deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv1/strided_slice_1/stack_2�
&decoder_output/deconv1/strided_slice_1StridedSlice%decoder_output/deconv1/stack:output:05decoder_output/deconv1/strided_slice_1/stack:output:07decoder_output/deconv1/strided_slice_1/stack_1:output:07decoder_output/deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/deconv1/strided_slice_1�
6decoder_output/deconv1/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_output_deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder_output/deconv1/conv2d_transpose/ReadVariableOp�
'decoder_output/deconv1/conv2d_transposeConv2DBackpropInput%decoder_output/deconv1/stack:output:0>decoder_output/deconv1/conv2d_transpose/ReadVariableOp:value:0)decoder_output/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2)
'decoder_output/deconv1/conv2d_transpose�
-decoder_output/deconv1/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder_output/deconv1/BiasAdd/ReadVariableOp�
decoder_output/deconv1/BiasAddBiasAdd0decoder_output/deconv1/conv2d_transpose:output:05decoder_output/deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 
decoder_output/deconv1/BiasAdd�
decoder_output/deconv1/ReluRelu'decoder_output/deconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
decoder_output/deconv1/Relu�
decoder_output/deconv2/ShapeShape)decoder_output/deconv1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_output/deconv2/Shape�
*decoder_output/deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_output/deconv2/strided_slice/stack�
,decoder_output/deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv2/strided_slice/stack_1�
,decoder_output/deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv2/strided_slice/stack_2�
$decoder_output/deconv2/strided_sliceStridedSlice%decoder_output/deconv2/Shape:output:03decoder_output/deconv2/strided_slice/stack:output:05decoder_output/deconv2/strided_slice/stack_1:output:05decoder_output/deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_output/deconv2/strided_slice�
decoder_output/deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/1�
decoder_output/deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/2�
decoder_output/deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/3�
decoder_output/deconv2/stackPack-decoder_output/deconv2/strided_slice:output:0'decoder_output/deconv2/stack/1:output:0'decoder_output/deconv2/stack/2:output:0'decoder_output/deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/deconv2/stack�
,decoder_output/deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/deconv2/strided_slice_1/stack�
.decoder_output/deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv2/strided_slice_1/stack_1�
.decoder_output/deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv2/strided_slice_1/stack_2�
&decoder_output/deconv2/strided_slice_1StridedSlice%decoder_output/deconv2/stack:output:05decoder_output/deconv2/strided_slice_1/stack:output:07decoder_output/deconv2/strided_slice_1/stack_1:output:07decoder_output/deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/deconv2/strided_slice_1�
6decoder_output/deconv2/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_output_deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder_output/deconv2/conv2d_transpose/ReadVariableOp�
'decoder_output/deconv2/conv2d_transposeConv2DBackpropInput%decoder_output/deconv2/stack:output:0>decoder_output/deconv2/conv2d_transpose/ReadVariableOp:value:0)decoder_output/deconv1/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2)
'decoder_output/deconv2/conv2d_transpose�
-decoder_output/deconv2/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder_output/deconv2/BiasAdd/ReadVariableOp�
decoder_output/deconv2/BiasAddBiasAdd0decoder_output/deconv2/conv2d_transpose:output:05decoder_output/deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 
decoder_output/deconv2/BiasAdd�
decoder_output/deconv2/ReluRelu'decoder_output/deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
decoder_output/deconv2/Relu�
decoder_output/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2 
decoder_output/flatten_3/Const�
 decoder_output/flatten_3/ReshapeReshape)decoder_output/deconv2/Relu:activations:0'decoder_output/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2"
 decoder_output/flatten_3/Reshape�
,decoder_output/dense_1/MatMul/ReadVariableOpReadVariableOp5decoder_output_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,decoder_output/dense_1/MatMul/ReadVariableOp�
decoder_output/dense_1/MatMulMatMul)decoder_output/flatten_3/Reshape:output:04decoder_output/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder_output/dense_1/MatMul�
-decoder_output/dense_1/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-decoder_output/dense_1/BiasAdd/ReadVariableOp�
decoder_output/dense_1/BiasAddBiasAdd'decoder_output/dense_1/MatMul:product:05decoder_output/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
decoder_output/dense_1/BiasAdd�
decoder_output/dense_1/SigmoidSigmoid'decoder_output/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
decoder_output/dense_1/SigmoidX
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_2/Shape�
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack�
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1�
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape�
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2
reshape_2/Reshape�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference"decoder_output/dense_1/Sigmoid:y:0inputs*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dreshape_2/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv1/Relu�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv2/Reluy
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Muls
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
flatten_2/Const�
flatten_2/ReshapeReshapeconv2/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2
flatten_2/Reshape�
z_log_var/MatMul/ReadVariableOpReadVariableOp7encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02!
z_log_var/MatMul/ReadVariableOp�
z_log_var/MatMulMatMulflatten_2/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/MatMul�
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp8encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/BiasAdd�
z_mean/MatMul/ReadVariableOpReadVariableOp4encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
z_mean/MatMul/ReadVariableOp�
z_mean/MatMulMatMulflatten_2/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/MatMul�
z_mean/BiasAdd/ReadVariableOpReadVariableOp5encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOp�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/BiasAdd{
tf.math.exp_1/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquarez_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
IdentityIdentity"decoder_output/dense_1/Sigmoid:y:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp.^decoder_output/deconv1/BiasAdd/ReadVariableOp7^decoder_output/deconv1/conv2d_transpose/ReadVariableOp.^decoder_output/deconv2/BiasAdd/ReadVariableOp7^decoder_output/deconv2/conv2d_transpose/ReadVariableOp.^decoder_output/dense_1/BiasAdd/ReadVariableOp-^decoder_output/dense_1/MatMul/ReadVariableOp,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#tf.math.reduce_mean_3/Mean:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp.^decoder_output/deconv1/BiasAdd/ReadVariableOp7^decoder_output/deconv1/conv2d_transpose/ReadVariableOp.^decoder_output/deconv2/BiasAdd/ReadVariableOp7^decoder_output/deconv2/conv2d_transpose/ReadVariableOp.^decoder_output/dense_1/BiasAdd/ReadVariableOp-^decoder_output/dense_1/MatMul/ReadVariableOp,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2^
-decoder_output/deconv1/BiasAdd/ReadVariableOp-decoder_output/deconv1/BiasAdd/ReadVariableOp2p
6decoder_output/deconv1/conv2d_transpose/ReadVariableOp6decoder_output/deconv1/conv2d_transpose/ReadVariableOp2^
-decoder_output/deconv2/BiasAdd/ReadVariableOp-decoder_output/deconv2/BiasAdd/ReadVariableOp2p
6decoder_output/deconv2/conv2d_transpose/ReadVariableOp6decoder_output/deconv2/conv2d_transpose/ReadVariableOp2^
-decoder_output/dense_1/BiasAdd/ReadVariableOp-decoder_output/dense_1/BiasAdd/ReadVariableOp2\
,decoder_output/dense_1/MatMul/ReadVariableOp,decoder_output/dense_1/MatMul/ReadVariableOp2Z
+encoder_output/conv1/BiasAdd/ReadVariableOp+encoder_output/conv1/BiasAdd/ReadVariableOp2X
*encoder_output/conv1/Conv2D/ReadVariableOp*encoder_output/conv1/Conv2D/ReadVariableOp2Z
+encoder_output/conv2/BiasAdd/ReadVariableOp+encoder_output/conv2/BiasAdd/ReadVariableOp2X
*encoder_output/conv2/Conv2D/ReadVariableOp*encoder_output/conv2/Conv2D/ReadVariableOp2b
/encoder_output/z_log_var/BiasAdd/ReadVariableOp/encoder_output/z_log_var/BiasAdd/ReadVariableOp2`
.encoder_output/z_log_var/MatMul/ReadVariableOp.encoder_output/z_log_var/MatMul/ReadVariableOp2\
,encoder_output/z_mean/BiasAdd/ReadVariableOp,encoder_output/z_mean/BiasAdd/ReadVariableOp2Z
+encoder_output/z_mean/MatMul/ReadVariableOp+encoder_output/z_mean/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_conv2_layer_call_and_return_conditional_losses_162824

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
{
&__inference_conv1_layer_call_fn_162813

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
k
"__inference_z_layer_call_fn_162945
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1611182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_161477

z_sampling
deconv1_161460
deconv1_161462
deconv2_161465
deconv2_161467
dense_1_161471
dense_1_161473
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCall
z_sampling*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1613902
reshape_3/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0deconv1_161460deconv1_161462*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv1_layer_call_and_return_conditional_losses_1613132!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_161465deconv2_161467*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv2_layer_call_and_return_conditional_losses_1613622!
deconv2/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1614202
flatten_3/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_1_161471dense_1_161473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1614392!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_161456

z_sampling
deconv1_161398
deconv1_161400
deconv2_161403
deconv2_161405
dense_1_161450
dense_1_161452
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCall
z_sampling*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1613902
reshape_3/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0deconv1_161398deconv1_161400*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv1_layer_call_and_return_conditional_losses_1613132!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_161403deconv2_161405*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv2_layer_call_and_return_conditional_losses_1613622!
deconv2/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1614202
flatten_3/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_1_161450dense_1_161452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1614392!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�	
�
$__inference_signature_wrapper_162048
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_1609322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
F
*__inference_flatten_3_layer_call_fn_162981

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1614202
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_162976

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
Reshape/shape/1�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:������������������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
l
=__inference_z_layer_call_and_return_conditional_losses_162913
inputs_0
inputs_1
identity�F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
random_normal/shape/1�
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2�>2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:���������	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������	2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:���������	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�	
�
B__inference_z_mean_layer_call_and_return_conditional_losses_162873

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�H
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161794
encoder_input
encoder_output_161726
encoder_output_161728
encoder_output_161730
encoder_output_161732
encoder_output_161734
encoder_output_161736
encoder_output_161738
encoder_output_161740
decoder_output_161745
decoder_output_161747
decoder_output_161749
decoder_output_161751
decoder_output_161753
decoder_output_161755
identity

identity_1��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCallencoder_inputencoder_output_161726encoder_output_161728encoder_output_161730encoder_output_161732encoder_output_161734encoder_output_161736encoder_output_161738encoder_output_161740*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1612512(
&encoder_output/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:2decoder_output_161745decoder_output_161747decoder_output_161749decoder_output_161751decoder_output_161753decoder_output_161755*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615392(
&decoder_output/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference/decoder_output/StatefulPartitionedCall:output:0encoder_input*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0encoder_output_161726encoder_output_161728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0encoder_output_161730encoder_output_161732*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Mul�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161738encoder_output_161740*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161734encoder_output_161736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
tf.math.exp_1/ExpExp*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquare'z_mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
add_loss_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_3/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_loss_1_layer_call_and_return_conditional_losses_1617112
add_loss_1/PartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#add_loss_1/PartitionedCall:output:1^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_161018

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������z2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������z2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_161197

inputs
conv1_161172
conv1_161174
conv2_161177
conv2_161179
z_mean_161183
z_mean_161185
z_log_var_161188
z_log_var_161190
identity

identity_1

identity_2��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1_161172conv1_161174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_161177conv2_161179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_mean_161183z_mean_161185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_log_var_161188z_log_var_161190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1610982
z/StatefulPartitionedCall�
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_conv1_layer_call_and_return_conditional_losses_162804

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
l
=__inference_z_layer_call_and_return_conditional_losses_162933
inputs_0
inputs_1
identity�F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
random_normal/shape/1�
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2���2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:���������	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������	2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:���������	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�
}
(__inference_deconv1_layer_call_fn_161323

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv1_layer_call_and_return_conditional_losses_1613132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�%
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_161251

inputs
conv1_161226
conv1_161228
conv2_161231
conv2_161233
z_mean_161237
z_mean_161239
z_log_var_161242
z_log_var_161244
identity

identity_1

identity_2��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1_161226conv1_161228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_161231conv2_161233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_mean_161237z_mean_161239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_log_var_161242z_log_var_161244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1611182
z/StatefulPartitionedCall�
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_dense_1_layer_call_fn_163001

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1614392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:������������������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
E__inference_z_log_var_layer_call_and_return_conditional_losses_161062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�%
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_161165
encoder_input
conv1_161140
conv1_161142
conv2_161145
conv2_161147
z_mean_161151
z_mean_161153
z_log_var_161156
z_log_var_161158
identity

identity_1

identity_2��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1_161140conv1_161142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_161145conv2_161147*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_mean_161151z_mean_161153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_log_var_161156z_log_var_161158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1611182
z/StatefulPartitionedCall�
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�N
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_162677

inputs4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��deconv1/BiasAdd/ReadVariableOp�'deconv1/conv2d_transpose/ReadVariableOp�deconv2/BiasAdd/ReadVariableOp�'deconv2/conv2d_transpose/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOpX
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_3/Shape�
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack�
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1�
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2x
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/3�
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape�
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_3/Reshapeh
deconv1/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
deconv1/Shape�
deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice/stack�
deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_1�
deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv1/strided_slice/stack_2�
deconv1/strided_sliceStridedSlicedeconv1/Shape:output:0$deconv1/strided_slice/stack:output:0&deconv1/strided_slice/stack_1:output:0&deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_sliced
deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/1d
deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/2d
deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv1/stack/3�
deconv1/stackPackdeconv1/strided_slice:output:0deconv1/stack/1:output:0deconv1/stack/2:output:0deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv1/stack�
deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv1/strided_slice_1/stack�
deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_1�
deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv1/strided_slice_1/stack_2�
deconv1/strided_slice_1StridedSlicedeconv1/stack:output:0&deconv1/strided_slice_1/stack:output:0(deconv1/strided_slice_1/stack_1:output:0(deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv1/strided_slice_1�
'deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv1/conv2d_transpose/ReadVariableOp�
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
deconv1/conv2d_transpose�
deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv1/BiasAdd/ReadVariableOp�
deconv1/BiasAddBiasAdd!deconv1/conv2d_transpose:output:0&deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
deconv1/BiasAddx
deconv1/ReluReludeconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
deconv1/Reluh
deconv2/ShapeShapedeconv1/Relu:activations:0*
T0*
_output_shapes
:2
deconv2/Shape�
deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice/stack�
deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_1�
deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
deconv2/strided_slice/stack_2�
deconv2/strided_sliceStridedSlicedeconv2/Shape:output:0$deconv2/strided_slice/stack:output:0&deconv2/strided_slice/stack_1:output:0&deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_sliced
deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/1d
deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/2d
deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
deconv2/stack/3�
deconv2/stackPackdeconv2/strided_slice:output:0deconv2/stack/1:output:0deconv2/stack/2:output:0deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
deconv2/stack�
deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
deconv2/strided_slice_1/stack�
deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_1�
deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
deconv2/strided_slice_1/stack_2�
deconv2/strided_slice_1StridedSlicedeconv2/stack:output:0&deconv2/strided_slice_1/stack:output:0(deconv2/strided_slice_1/stack_1:output:0(deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
deconv2/strided_slice_1�
'deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02)
'deconv2/conv2d_transpose/ReadVariableOp�
deconv2/conv2d_transposeConv2DBackpropInputdeconv2/stack:output:0/deconv2/conv2d_transpose/ReadVariableOp:value:0deconv1/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
deconv2/conv2d_transpose�
deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
deconv2/BiasAdd/ReadVariableOp�
deconv2/BiasAddBiasAdd!deconv2/conv2d_transpose:output:0&deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
deconv2/BiasAddx
deconv2/ReluReludeconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
deconv2/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_3/Const�
flatten_3/ReshapeReshapedeconv2/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_3/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddz
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Sigmoid�
IdentityIdentitydense_1/Sigmoid:y:0^deconv1/BiasAdd/ReadVariableOp(^deconv1/conv2d_transpose/ReadVariableOp^deconv2/BiasAdd/ReadVariableOp(^deconv2/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2@
deconv1/BiasAdd/ReadVariableOpdeconv1/BiasAdd/ReadVariableOp2R
'deconv1/conv2d_transpose/ReadVariableOp'deconv1/conv2d_transpose/ReadVariableOp2@
deconv2/BiasAdd/ReadVariableOpdeconv2/BiasAdd/ReadVariableOp2R
'deconv2/conv2d_transpose/ReadVariableOp'deconv2/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
A__inference_conv2_layer_call_and_return_conditional_losses_160996

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_162788

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������  2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_encoder_output_layer_call_fn_162614

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1612512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�G
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161868

inputs
encoder_output_161800
encoder_output_161802
encoder_output_161804
encoder_output_161806
encoder_output_161808
encoder_output_161810
encoder_output_161812
encoder_output_161814
decoder_output_161819
decoder_output_161821
decoder_output_161823
decoder_output_161825
decoder_output_161827
decoder_output_161829
identity

identity_1��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_output_161800encoder_output_161802encoder_output_161804encoder_output_161806encoder_output_161808encoder_output_161810encoder_output_161812encoder_output_161814*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1611972(
&encoder_output/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:2decoder_output_161819decoder_output_161821decoder_output_161823decoder_output_161825decoder_output_161827decoder_output_161829*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615012(
&decoder_output/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference/decoder_output/StatefulPartitionedCall:output:0inputs*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0encoder_output_161800encoder_output_161802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0encoder_output_161804encoder_output_161806*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Mul�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161812encoder_output_161814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161808encoder_output_161810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
tf.math.exp_1/ExpExp*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquare'z_mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
add_loss_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_3/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_loss_1_layer_call_and_return_conditional_losses_1617112
add_loss_1/PartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#add_loss_1/PartitionedCall:output:1^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
"__inference_z_layer_call_fn_162939
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1610982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������	
"
_user_specified_name
inputs/1
�

�
(__inference_vae_cnn_layer_call_fn_162005
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:����������: *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_vae_cnn_layer_call_and_return_conditional_losses_1619732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
�
/__inference_decoder_output_layer_call_fn_162774

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_161420

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
Reshape/shape/1�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:������������������2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_161439

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�

*__inference_z_log_var_layer_call_fn_162863

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�
{
&__inference_conv2_layer_call_fn_162833

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_163328
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias'
#assignvariableop_4_z_log_var_kernel%
!assignvariableop_5_z_log_var_bias$
 assignvariableop_6_z_mean_kernel"
assignvariableop_7_z_mean_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate&
"assignvariableop_13_deconv1_kernel$
 assignvariableop_14_deconv1_bias&
"assignvariableop_15_deconv2_kernel$
 assignvariableop_16_deconv2_bias&
"assignvariableop_17_dense_1_kernel$
 assignvariableop_18_dense_1_bias
assignvariableop_19_total
assignvariableop_20_count+
'assignvariableop_21_adam_conv1_kernel_m)
%assignvariableop_22_adam_conv1_bias_m+
'assignvariableop_23_adam_conv2_kernel_m)
%assignvariableop_24_adam_conv2_bias_m/
+assignvariableop_25_adam_z_log_var_kernel_m-
)assignvariableop_26_adam_z_log_var_bias_m,
(assignvariableop_27_adam_z_mean_kernel_m*
&assignvariableop_28_adam_z_mean_bias_m-
)assignvariableop_29_adam_deconv1_kernel_m+
'assignvariableop_30_adam_deconv1_bias_m-
)assignvariableop_31_adam_deconv2_kernel_m+
'assignvariableop_32_adam_deconv2_bias_m-
)assignvariableop_33_adam_dense_1_kernel_m+
'assignvariableop_34_adam_dense_1_bias_m+
'assignvariableop_35_adam_conv1_kernel_v)
%assignvariableop_36_adam_conv1_bias_v+
'assignvariableop_37_adam_conv2_kernel_v)
%assignvariableop_38_adam_conv2_bias_v/
+assignvariableop_39_adam_z_log_var_kernel_v-
)assignvariableop_40_adam_z_log_var_bias_v,
(assignvariableop_41_adam_z_mean_kernel_v*
&assignvariableop_42_adam_z_mean_bias_v-
)assignvariableop_43_adam_deconv1_kernel_v+
'assignvariableop_44_adam_deconv1_bias_v-
)assignvariableop_45_adam_deconv2_kernel_v+
'assignvariableop_46_adam_deconv2_bias_v-
)assignvariableop_47_adam_dense_1_kernel_v+
'assignvariableop_48_adam_dense_1_bias_v
identity_50��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_z_log_var_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_z_log_var_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_deconv1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp assignvariableop_14_deconv1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_deconv2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp assignvariableop_16_deconv2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_conv1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_conv1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_z_log_var_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_z_log_var_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_z_mean_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_z_mean_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_deconv1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_deconv1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_deconv2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_deconv2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_conv1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_conv1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_conv2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_conv2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_z_log_var_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_z_log_var_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_z_mean_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_z_mean_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_deconv1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_deconv1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_deconv2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_deconv2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49�	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_162839

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������z2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������z2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_161539

inputs
deconv1_161522
deconv1_161524
deconv2_161527
deconv2_161529
dense_1_161533
dense_1_161535
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1613902
reshape_3/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0deconv1_161522deconv1_161524*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv1_layer_call_and_return_conditional_losses_1613132!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_161527deconv2_161529*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv2_layer_call_and_return_conditional_losses_1613622!
deconv2/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1614202
flatten_3/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_1_161533dense_1_161535*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1614392!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
/__inference_encoder_output_layer_call_fn_161274
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1612512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
r
F__inference_add_loss_1_layer_call_and_return_conditional_losses_162887

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
F
*__inference_reshape_2_layer_call_fn_162793

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
=__inference_z_layer_call_and_return_conditional_losses_161118

inputs
inputs_1
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
random_normal/shape/1�
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2��<2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:���������	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������	2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:���������	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�b
�
__inference__traced_save_163171
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop-
)savev2_deconv1_kernel_read_readvariableop+
'savev2_deconv1_bias_read_readvariableop-
)savev2_deconv2_kernel_read_readvariableop+
'savev2_deconv2_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop6
2savev2_adam_z_log_var_kernel_m_read_readvariableop4
0savev2_adam_z_log_var_bias_m_read_readvariableop3
/savev2_adam_z_mean_kernel_m_read_readvariableop1
-savev2_adam_z_mean_bias_m_read_readvariableop4
0savev2_adam_deconv1_kernel_m_read_readvariableop2
.savev2_adam_deconv1_bias_m_read_readvariableop4
0savev2_adam_deconv2_kernel_m_read_readvariableop2
.savev2_adam_deconv2_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop6
2savev2_adam_z_log_var_kernel_v_read_readvariableop4
0savev2_adam_z_log_var_bias_v_read_readvariableop3
/savev2_adam_z_mean_kernel_v_read_readvariableop1
-savev2_adam_z_mean_bias_v_read_readvariableop4
0savev2_adam_deconv1_kernel_v_read_readvariableop2
.savev2_adam_deconv1_bias_v_read_readvariableop4
0savev2_adam_deconv2_kernel_v_read_readvariableop2
.savev2_adam_deconv2_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*�
value�B�2B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop)savev2_deconv1_kernel_read_readvariableop'savev2_deconv1_bias_read_readvariableop)savev2_deconv2_kernel_read_readvariableop'savev2_deconv2_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop2savev2_adam_z_log_var_kernel_m_read_readvariableop0savev2_adam_z_log_var_bias_m_read_readvariableop/savev2_adam_z_mean_kernel_m_read_readvariableop-savev2_adam_z_mean_bias_m_read_readvariableop0savev2_adam_deconv1_kernel_m_read_readvariableop.savev2_adam_deconv1_bias_m_read_readvariableop0savev2_adam_deconv2_kernel_m_read_readvariableop.savev2_adam_deconv2_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop2savev2_adam_z_log_var_kernel_v_read_readvariableop0savev2_adam_z_log_var_bias_v_read_readvariableop/savev2_adam_z_mean_kernel_v_read_readvariableop-savev2_adam_z_mean_bias_v_read_readvariableop0savev2_adam_deconv1_kernel_v_read_readvariableop.savev2_adam_deconv1_bias_v_read_readvariableop0savev2_adam_deconv2_kernel_v_read_readvariableop.savev2_adam_deconv2_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::	�z	:	:	�z	:	: : : : : :::::
��:�: : :::::	�z	:	:	�z	:	:::::
��:�:::::	�z	:	:	�z	:	:::::
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�z	: 

_output_shapes
:	:%!

_output_shapes
:	�z	: 

_output_shapes
:	:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�z	: 

_output_shapes
:	:%!

_output_shapes
:	�z	: 

_output_shapes
:	:,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::%(!

_output_shapes
:	�z	: )

_output_shapes
:	:%*!

_output_shapes
:	�z	: +

_output_shapes
:	:,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::&0"
 
_output_shapes
:
��:!1

_output_shapes	
:�:2

_output_shapes
: 
�
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_161390

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������	:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162213

inputs7
3encoder_output_conv1_conv2d_readvariableop_resource8
4encoder_output_conv1_biasadd_readvariableop_resource7
3encoder_output_conv2_conv2d_readvariableop_resource8
4encoder_output_conv2_biasadd_readvariableop_resource8
4encoder_output_z_mean_matmul_readvariableop_resource9
5encoder_output_z_mean_biasadd_readvariableop_resource;
7encoder_output_z_log_var_matmul_readvariableop_resource<
8encoder_output_z_log_var_biasadd_readvariableop_resourceC
?decoder_output_deconv1_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv1_biasadd_readvariableop_resourceC
?decoder_output_deconv2_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv2_biasadd_readvariableop_resource9
5decoder_output_dense_1_matmul_readvariableop_resource:
6decoder_output_dense_1_biasadd_readvariableop_resource
identity

identity_1��conv1/BiasAdd/ReadVariableOp�conv1/Conv2D/ReadVariableOp�conv2/BiasAdd/ReadVariableOp�conv2/Conv2D/ReadVariableOp�-decoder_output/deconv1/BiasAdd/ReadVariableOp�6decoder_output/deconv1/conv2d_transpose/ReadVariableOp�-decoder_output/deconv2/BiasAdd/ReadVariableOp�6decoder_output/deconv2/conv2d_transpose/ReadVariableOp�-decoder_output/dense_1/BiasAdd/ReadVariableOp�,decoder_output/dense_1/MatMul/ReadVariableOp�+encoder_output/conv1/BiasAdd/ReadVariableOp�*encoder_output/conv1/Conv2D/ReadVariableOp�+encoder_output/conv2/BiasAdd/ReadVariableOp�*encoder_output/conv2/Conv2D/ReadVariableOp�/encoder_output/z_log_var/BiasAdd/ReadVariableOp�.encoder_output/z_log_var/MatMul/ReadVariableOp�,encoder_output/z_mean/BiasAdd/ReadVariableOp�+encoder_output/z_mean/MatMul/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpv
encoder_output/reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2 
encoder_output/reshape_2/Shape�
,encoder_output/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,encoder_output/reshape_2/strided_slice/stack�
.encoder_output/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.encoder_output/reshape_2/strided_slice/stack_1�
.encoder_output/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.encoder_output/reshape_2/strided_slice/stack_2�
&encoder_output/reshape_2/strided_sliceStridedSlice'encoder_output/reshape_2/Shape:output:05encoder_output/reshape_2/strided_slice/stack:output:07encoder_output/reshape_2/strided_slice/stack_1:output:07encoder_output/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&encoder_output/reshape_2/strided_slice�
(encoder_output/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder_output/reshape_2/Reshape/shape/1�
(encoder_output/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder_output/reshape_2/Reshape/shape/2�
(encoder_output/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(encoder_output/reshape_2/Reshape/shape/3�
&encoder_output/reshape_2/Reshape/shapePack/encoder_output/reshape_2/strided_slice:output:01encoder_output/reshape_2/Reshape/shape/1:output:01encoder_output/reshape_2/Reshape/shape/2:output:01encoder_output/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&encoder_output/reshape_2/Reshape/shape�
 encoder_output/reshape_2/ReshapeReshapeinputs/encoder_output/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2"
 encoder_output/reshape_2/Reshape�
*encoder_output/conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv1/Conv2D/ReadVariableOp�
encoder_output/conv1/Conv2DConv2D)encoder_output/reshape_2/Reshape:output:02encoder_output/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
encoder_output/conv1/Conv2D�
+encoder_output/conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv1/BiasAdd/ReadVariableOp�
encoder_output/conv1/BiasAddBiasAdd$encoder_output/conv1/Conv2D:output:03encoder_output/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
encoder_output/conv1/BiasAdd�
encoder_output/conv1/ReluRelu%encoder_output/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
encoder_output/conv1/Relu�
*encoder_output/conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv2/Conv2D/ReadVariableOp�
encoder_output/conv2/Conv2DConv2D'encoder_output/conv1/Relu:activations:02encoder_output/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
encoder_output/conv2/Conv2D�
+encoder_output/conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv2/BiasAdd/ReadVariableOp�
encoder_output/conv2/BiasAddBiasAdd$encoder_output/conv2/Conv2D:output:03encoder_output/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
encoder_output/conv2/BiasAdd�
encoder_output/conv2/ReluRelu%encoder_output/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
encoder_output/conv2/Relu�
encoder_output/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2 
encoder_output/flatten_2/Const�
 encoder_output/flatten_2/ReshapeReshape'encoder_output/conv2/Relu:activations:0'encoder_output/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2"
 encoder_output/flatten_2/Reshape�
+encoder_output/z_mean/MatMul/ReadVariableOpReadVariableOp4encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02-
+encoder_output/z_mean/MatMul/ReadVariableOp�
encoder_output/z_mean/MatMulMatMul)encoder_output/flatten_2/Reshape:output:03encoder_output/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
encoder_output/z_mean/MatMul�
,encoder_output/z_mean/BiasAdd/ReadVariableOpReadVariableOp5encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02.
,encoder_output/z_mean/BiasAdd/ReadVariableOp�
encoder_output/z_mean/BiasAddBiasAdd&encoder_output/z_mean/MatMul:product:04encoder_output/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
encoder_output/z_mean/BiasAdd�
.encoder_output/z_log_var/MatMul/ReadVariableOpReadVariableOp7encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype020
.encoder_output/z_log_var/MatMul/ReadVariableOp�
encoder_output/z_log_var/MatMulMatMul)encoder_output/flatten_2/Reshape:output:06encoder_output/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2!
encoder_output/z_log_var/MatMul�
/encoder_output/z_log_var/BiasAdd/ReadVariableOpReadVariableOp8encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype021
/encoder_output/z_log_var/BiasAdd/ReadVariableOp�
 encoder_output/z_log_var/BiasAddBiasAdd)encoder_output/z_log_var/MatMul:product:07encoder_output/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2"
 encoder_output/z_log_var/BiasAdd�
encoder_output/z/ShapeShape&encoder_output/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder_output/z/Shape�
$encoder_output/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$encoder_output/z/strided_slice/stack�
&encoder_output/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_1�
&encoder_output/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_2�
encoder_output/z/strided_sliceStridedSliceencoder_output/z/Shape:output:0-encoder_output/z/strided_slice/stack:output:0/encoder_output/z/strided_slice/stack_1:output:0/encoder_output/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
encoder_output/z/strided_slice�
&encoder_output/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2(
&encoder_output/z/random_normal/shape/1�
$encoder_output/z/random_normal/shapePack'encoder_output/z/strided_slice:output:0/encoder_output/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$encoder_output/z/random_normal/shape�
#encoder_output/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder_output/z/random_normal/mean�
%encoder_output/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%encoder_output/z/random_normal/stddev�
3encoder_output/z/random_normal/RandomStandardNormalRandomStandardNormal-encoder_output/z/random_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2���25
3encoder_output/z/random_normal/RandomStandardNormal�
"encoder_output/z/random_normal/mulMul<encoder_output/z/random_normal/RandomStandardNormal:output:0.encoder_output/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2$
"encoder_output/z/random_normal/mul�
encoder_output/z/random_normalAdd&encoder_output/z/random_normal/mul:z:0,encoder_output/z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2 
encoder_output/z/random_normal�
encoder_output/z/ExpExp)encoder_output/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/Exp�
encoder_output/z/mulMulencoder_output/z/Exp:y:0"encoder_output/z/random_normal:z:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/mul�
encoder_output/z/addAddV2&encoder_output/z_mean/BiasAdd:output:0encoder_output/z/mul:z:0*
T0*'
_output_shapes
:���������	2
encoder_output/z/add�
decoder_output/reshape_3/ShapeShapeencoder_output/z/add:z:0*
T0*
_output_shapes
:2 
decoder_output/reshape_3/Shape�
,decoder_output/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/reshape_3/strided_slice/stack�
.decoder_output/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_3/strided_slice/stack_1�
.decoder_output/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_3/strided_slice/stack_2�
&decoder_output/reshape_3/strided_sliceStridedSlice'decoder_output/reshape_3/Shape:output:05decoder_output/reshape_3/strided_slice/stack:output:07decoder_output/reshape_3/strided_slice/stack_1:output:07decoder_output/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/reshape_3/strided_slice�
(decoder_output/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/1�
(decoder_output/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/2�
(decoder_output/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_3/Reshape/shape/3�
&decoder_output/reshape_3/Reshape/shapePack/decoder_output/reshape_3/strided_slice:output:01decoder_output/reshape_3/Reshape/shape/1:output:01decoder_output/reshape_3/Reshape/shape/2:output:01decoder_output/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&decoder_output/reshape_3/Reshape/shape�
 decoder_output/reshape_3/ReshapeReshapeencoder_output/z/add:z:0/decoder_output/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2"
 decoder_output/reshape_3/Reshape�
decoder_output/deconv1/ShapeShape)decoder_output/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2
decoder_output/deconv1/Shape�
*decoder_output/deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_output/deconv1/strided_slice/stack�
,decoder_output/deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv1/strided_slice/stack_1�
,decoder_output/deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv1/strided_slice/stack_2�
$decoder_output/deconv1/strided_sliceStridedSlice%decoder_output/deconv1/Shape:output:03decoder_output/deconv1/strided_slice/stack:output:05decoder_output/deconv1/strided_slice/stack_1:output:05decoder_output/deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_output/deconv1/strided_slice�
decoder_output/deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/1�
decoder_output/deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/2�
decoder_output/deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv1/stack/3�
decoder_output/deconv1/stackPack-decoder_output/deconv1/strided_slice:output:0'decoder_output/deconv1/stack/1:output:0'decoder_output/deconv1/stack/2:output:0'decoder_output/deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/deconv1/stack�
,decoder_output/deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/deconv1/strided_slice_1/stack�
.decoder_output/deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv1/strided_slice_1/stack_1�
.decoder_output/deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv1/strided_slice_1/stack_2�
&decoder_output/deconv1/strided_slice_1StridedSlice%decoder_output/deconv1/stack:output:05decoder_output/deconv1/strided_slice_1/stack:output:07decoder_output/deconv1/strided_slice_1/stack_1:output:07decoder_output/deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/deconv1/strided_slice_1�
6decoder_output/deconv1/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_output_deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder_output/deconv1/conv2d_transpose/ReadVariableOp�
'decoder_output/deconv1/conv2d_transposeConv2DBackpropInput%decoder_output/deconv1/stack:output:0>decoder_output/deconv1/conv2d_transpose/ReadVariableOp:value:0)decoder_output/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2)
'decoder_output/deconv1/conv2d_transpose�
-decoder_output/deconv1/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder_output/deconv1/BiasAdd/ReadVariableOp�
decoder_output/deconv1/BiasAddBiasAdd0decoder_output/deconv1/conv2d_transpose:output:05decoder_output/deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 
decoder_output/deconv1/BiasAdd�
decoder_output/deconv1/ReluRelu'decoder_output/deconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
decoder_output/deconv1/Relu�
decoder_output/deconv2/ShapeShape)decoder_output/deconv1/Relu:activations:0*
T0*
_output_shapes
:2
decoder_output/deconv2/Shape�
*decoder_output/deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*decoder_output/deconv2/strided_slice/stack�
,decoder_output/deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv2/strided_slice/stack_1�
,decoder_output/deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,decoder_output/deconv2/strided_slice/stack_2�
$decoder_output/deconv2/strided_sliceStridedSlice%decoder_output/deconv2/Shape:output:03decoder_output/deconv2/strided_slice/stack:output:05decoder_output/deconv2/strided_slice/stack_1:output:05decoder_output/deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$decoder_output/deconv2/strided_slice�
decoder_output/deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/1�
decoder_output/deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/2�
decoder_output/deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
decoder_output/deconv2/stack/3�
decoder_output/deconv2/stackPack-decoder_output/deconv2/strided_slice:output:0'decoder_output/deconv2/stack/1:output:0'decoder_output/deconv2/stack/2:output:0'decoder_output/deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
decoder_output/deconv2/stack�
,decoder_output/deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/deconv2/strided_slice_1/stack�
.decoder_output/deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv2/strided_slice_1/stack_1�
.decoder_output/deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/deconv2/strided_slice_1/stack_2�
&decoder_output/deconv2/strided_slice_1StridedSlice%decoder_output/deconv2/stack:output:05decoder_output/deconv2/strided_slice_1/stack:output:07decoder_output/deconv2/strided_slice_1/stack_1:output:07decoder_output/deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/deconv2/strided_slice_1�
6decoder_output/deconv2/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_output_deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype028
6decoder_output/deconv2/conv2d_transpose/ReadVariableOp�
'decoder_output/deconv2/conv2d_transposeConv2DBackpropInput%decoder_output/deconv2/stack:output:0>decoder_output/deconv2/conv2d_transpose/ReadVariableOp:value:0)decoder_output/deconv1/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2)
'decoder_output/deconv2/conv2d_transpose�
-decoder_output/deconv2/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder_output/deconv2/BiasAdd/ReadVariableOp�
decoder_output/deconv2/BiasAddBiasAdd0decoder_output/deconv2/conv2d_transpose:output:05decoder_output/deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 
decoder_output/deconv2/BiasAdd�
decoder_output/deconv2/ReluRelu'decoder_output/deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
decoder_output/deconv2/Relu�
decoder_output/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2 
decoder_output/flatten_3/Const�
 decoder_output/flatten_3/ReshapeReshape)decoder_output/deconv2/Relu:activations:0'decoder_output/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2"
 decoder_output/flatten_3/Reshape�
,decoder_output/dense_1/MatMul/ReadVariableOpReadVariableOp5decoder_output_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,decoder_output/dense_1/MatMul/ReadVariableOp�
decoder_output/dense_1/MatMulMatMul)decoder_output/flatten_3/Reshape:output:04decoder_output/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder_output/dense_1/MatMul�
-decoder_output/dense_1/BiasAdd/ReadVariableOpReadVariableOp6decoder_output_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-decoder_output/dense_1/BiasAdd/ReadVariableOp�
decoder_output/dense_1/BiasAddBiasAdd'decoder_output/dense_1/MatMul:product:05decoder_output/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
decoder_output/dense_1/BiasAdd�
decoder_output/dense_1/SigmoidSigmoid'decoder_output/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
decoder_output/dense_1/SigmoidX
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_2/Shape�
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack�
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1�
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape�
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2
reshape_2/Reshape�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference"decoder_output/dense_1/Sigmoid:y:0inputs*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dreshape_2/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv1/Relu�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv2/Reluy
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Muls
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
flatten_2/Const�
flatten_2/ReshapeReshapeconv2/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2
flatten_2/Reshape�
z_log_var/MatMul/ReadVariableOpReadVariableOp7encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02!
z_log_var/MatMul/ReadVariableOp�
z_log_var/MatMulMatMulflatten_2/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/MatMul�
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp8encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/BiasAdd�
z_mean/MatMul/ReadVariableOpReadVariableOp4encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
z_mean/MatMul/ReadVariableOp�
z_mean/MatMulMatMulflatten_2/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/MatMul�
z_mean/BiasAdd/ReadVariableOpReadVariableOp5encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOp�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/BiasAdd{
tf.math.exp_1/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquarez_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
IdentityIdentity"decoder_output/dense_1/Sigmoid:y:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp.^decoder_output/deconv1/BiasAdd/ReadVariableOp7^decoder_output/deconv1/conv2d_transpose/ReadVariableOp.^decoder_output/deconv2/BiasAdd/ReadVariableOp7^decoder_output/deconv2/conv2d_transpose/ReadVariableOp.^decoder_output/dense_1/BiasAdd/ReadVariableOp-^decoder_output/dense_1/MatMul/ReadVariableOp,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#tf.math.reduce_mean_3/Mean:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp.^decoder_output/deconv1/BiasAdd/ReadVariableOp7^decoder_output/deconv1/conv2d_transpose/ReadVariableOp.^decoder_output/deconv2/BiasAdd/ReadVariableOp7^decoder_output/deconv2/conv2d_transpose/ReadVariableOp.^decoder_output/dense_1/BiasAdd/ReadVariableOp-^decoder_output/dense_1/MatMul/ReadVariableOp,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2^
-decoder_output/deconv1/BiasAdd/ReadVariableOp-decoder_output/deconv1/BiasAdd/ReadVariableOp2p
6decoder_output/deconv1/conv2d_transpose/ReadVariableOp6decoder_output/deconv1/conv2d_transpose/ReadVariableOp2^
-decoder_output/deconv2/BiasAdd/ReadVariableOp-decoder_output/deconv2/BiasAdd/ReadVariableOp2p
6decoder_output/deconv2/conv2d_transpose/ReadVariableOp6decoder_output/deconv2/conv2d_transpose/ReadVariableOp2^
-decoder_output/dense_1/BiasAdd/ReadVariableOp-decoder_output/dense_1/BiasAdd/ReadVariableOp2\
,decoder_output/dense_1/MatMul/ReadVariableOp,decoder_output/dense_1/MatMul/ReadVariableOp2Z
+encoder_output/conv1/BiasAdd/ReadVariableOp+encoder_output/conv1/BiasAdd/ReadVariableOp2X
*encoder_output/conv1/Conv2D/ReadVariableOp*encoder_output/conv1/Conv2D/ReadVariableOp2Z
+encoder_output/conv2/BiasAdd/ReadVariableOp+encoder_output/conv2/BiasAdd/ReadVariableOp2X
*encoder_output/conv2/Conv2D/ReadVariableOp*encoder_output/conv2/Conv2D/ReadVariableOp2b
/encoder_output/z_log_var/BiasAdd/ReadVariableOp/encoder_output/z_log_var/BiasAdd/ReadVariableOp2`
.encoder_output/z_log_var/MatMul/ReadVariableOp.encoder_output/z_log_var/MatMul/ReadVariableOp2\
,encoder_output/z_mean/BiasAdd/ReadVariableOp,encoder_output/z_mean/BiasAdd/ReadVariableOp2Z
+encoder_output/z_mean/MatMul/ReadVariableOp+encoder_output/z_mean/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_add_loss_1_layer_call_fn_162893

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_loss_1_layer_call_and_return_conditional_losses_1617112
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�H
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161723
encoder_input
encoder_output_161608
encoder_output_161610
encoder_output_161612
encoder_output_161614
encoder_output_161616
encoder_output_161618
encoder_output_161620
encoder_output_161622
decoder_output_161661
decoder_output_161663
decoder_output_161665
decoder_output_161667
decoder_output_161669
decoder_output_161671
identity

identity_1��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCallencoder_inputencoder_output_161608encoder_output_161610encoder_output_161612encoder_output_161614encoder_output_161616encoder_output_161618encoder_output_161620encoder_output_161622*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1611972(
&encoder_output/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:2decoder_output_161661decoder_output_161663decoder_output_161665decoder_output_161667decoder_output_161669decoder_output_161671*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615012(
&decoder_output/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference/decoder_output/StatefulPartitionedCall:output:0encoder_input*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0encoder_output_161608encoder_output_161610*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0encoder_output_161612encoder_output_161614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Mul�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161620encoder_output_161622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161616encoder_output_161618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
tf.math.exp_1/ExpExp*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquare'z_mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
add_loss_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_3/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_loss_1_layer_call_and_return_conditional_losses_1617112
add_loss_1/PartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#add_loss_1/PartitionedCall:output:1^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�J
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_162505

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2��conv1/BiasAdd/ReadVariableOp�conv1/Conv2D/ReadVariableOp�conv2/BiasAdd/ReadVariableOp�conv2/Conv2D/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpX
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_2/Shape�
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack�
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1�
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape�
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2
reshape_2/Reshape�
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dreshape_2/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv1/Relu�
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv2/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
flatten_2/Const�
flatten_2/ReshapeReshapeconv2/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2
flatten_2/Reshape�
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
z_mean/MatMul/ReadVariableOp�
z_mean/MatMulMatMulflatten_2/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/MatMul�
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOp�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/BiasAdd�
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02!
z_log_var/MatMul/ReadVariableOp�
z_log_var/MatMulMatMulflatten_2/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/MatMul�
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/BiasAddY
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2	
z/Shapex
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
z/strided_slice/stack|
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
z/strided_slice/stack_1|
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
z/strided_slice/stack_2�
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
z/strided_slicet
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
z/random_normal/shape/1�
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
z/random_normal/shapeq
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
z/random_normal/meanu
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
z/random_normal/stddev�
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2⽴2&
$z/random_normal/RandomStandardNormal�
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
z/random_normal/mul�
z/random_normalAddz/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
z/random_normalc
z/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
z/Expg
z/mulMul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:���������	2
z/mulm
z/addAddV2z_mean/BiasAdd:output:0	z/mul:z:0*
T0*'
_output_shapes
:���������	2
z/add�
IdentityIdentityz_mean/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identityz_log_var/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity	z/add:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_162992

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�%
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_161136
encoder_input
conv1_160980
conv1_160982
conv2_161007
conv2_161009
z_mean_161047
z_mean_161049
z_log_var_161073
z_log_var_161075
identity

identity_1

identity_2��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0conv1_160980conv1_160982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_161007conv2_161009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_mean_161047z_mean_161049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0z_log_var_161073z_log_var_161075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1610982
z/StatefulPartitionedCall�
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_160950

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������  2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_flatten_2_layer_call_fn_162844

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������z2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_conv1_layer_call_and_return_conditional_losses_160969

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
/__inference_encoder_output_layer_call_fn_162589

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1611972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�G
�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161973

inputs
encoder_output_161905
encoder_output_161907
encoder_output_161909
encoder_output_161911
encoder_output_161913
encoder_output_161915
encoder_output_161917
encoder_output_161919
decoder_output_161924
decoder_output_161926
decoder_output_161928
decoder_output_161930
decoder_output_161932
decoder_output_161934
identity

identity_1��conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�&encoder_output/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
&encoder_output/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_output_161905encoder_output_161907encoder_output_161909encoder_output_161911encoder_output_161913encoder_output_161915encoder_output_161917encoder_output_161919*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1612512(
&encoder_output/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:2decoder_output_161924decoder_output_161926decoder_output_161928decoder_output_161930decoder_output_161932decoder_output_161934*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615392(
&decoder_output/StatefulPartitionedCall�
reshape_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_1609502
reshape_2/PartitionedCall�
.tf.math.squared_difference_1/SquaredDifferenceSquaredDifference/decoder_output/StatefulPartitionedCall:output:0inputs*
T0*(
_output_shapes
:����������20
.tf.math.squared_difference_1/SquaredDifference�
conv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0encoder_output_161905encoder_output_161907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1609692
conv1/StatefulPartitionedCall�
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2.
,tf.math.reduce_mean_2/Mean/reduction_indices�
tf.math.reduce_mean_2/MeanMean2tf.math.squared_difference_1/SquaredDifference:z:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_mean_2/Mean�
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0encoder_output_161909encoder_output_161911*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1609962
conv2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.multiply_2/Mul/y�
tf.math.multiply_2/MulMul#tf.math.reduce_mean_2/Mean:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_2/Mul�
flatten_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1610182
flatten_2/PartitionedCall�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161917encoder_output_161919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1610622#
!z_log_var/StatefulPartitionedCall�
z_mean/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0encoder_output_161913encoder_output_161915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362 
z_mean/StatefulPartitionedCall�
tf.math.exp_1/ExpExp*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.exp_1/Exp�
tf.math.square_1/SquareSquare'z_mean/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.square_1/Square�
tf.__operators__.add_2/AddV2AddV2tf.math.exp_1/Exp:y:0tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2
tf.__operators__.add_2/AddV2�
tf.math.subtract_2/SubSub tf.__operators__.add_2/AddV2:z:0*z_log_var/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_2/Suby
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
tf.math.subtract_3/Sub/y�
tf.math.subtract_3/SubSubtf.math.subtract_2/Sub:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2
tf.math.subtract_3/Sub�
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2,
*tf.math.reduce_sum_1/Sum/reduction_indices�
tf.math.reduce_sum_1/SumSumtf.math.subtract_3/Sub:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2
tf.math.reduce_sum_1/Sumy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
tf.math.multiply_3/Mul/y�
tf.math.multiply_3/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2
tf.math.multiply_3/Mul�
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2
tf.__operators__.add_3/AddV2�
tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
tf.math.reduce_mean_3/Const�
tf.math.reduce_mean_3/MeanMean tf.__operators__.add_3/AddV2:z:0$tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2
tf.math.reduce_mean_3/Mean�
add_loss_1/PartitionedCallPartitionedCall#tf.math.reduce_mean_3/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_add_loss_1_layer_call_and_return_conditional_losses_1617112
add_loss_1/PartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity#add_loss_1/PartitionedCall:output:1^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*_
_input_shapesN
L:����������::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
=__inference_z_layer_call_and_return_conditional_losses_161098

inputs
inputs_1
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicep
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
random_normal/shape/1�
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2��22$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:���������	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������	2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:���������	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������	:���������	:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
/__inference_encoder_output_layer_call_fn_161220
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������	:���������	:���������	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_encoder_output_layer_call_and_return_conditional_losses_1611972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
|
'__inference_z_mean_layer_call_fn_162882

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1610362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�
F
*__inference_reshape_3_layer_call_fn_162964

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1613902
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������	:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
(__inference_vae_cnn_layer_call_fn_162446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:����������: *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_vae_cnn_layer_call_and_return_conditional_losses_1619732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
(__inference_vae_cnn_layer_call_fn_162412

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:����������: *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_vae_cnn_layer_call_and_return_conditional_losses_1618682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_decoder_output_layer_call_fn_161516

z_sampling
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
z_samplingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�
�
/__inference_decoder_output_layer_call_fn_161554

z_sampling
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
z_samplingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�J
�
J__inference_encoder_output_layer_call_and_return_conditional_losses_162564

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2��conv1/BiasAdd/ReadVariableOp�conv1/Conv2D/ReadVariableOp�conv2/BiasAdd/ReadVariableOp�conv2/Conv2D/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpX
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_2/Shape�
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack�
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1�
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3�
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape�
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2
reshape_2/Reshape�
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dreshape_2/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv1/Relu�
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2

conv2/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
flatten_2/Const�
flatten_2/ReshapeReshapeconv2/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2
flatten_2/Reshape�
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
z_mean/MatMul/ReadVariableOp�
z_mean/MatMulMatMulflatten_2/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/MatMul�
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOp�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_mean/BiasAdd�
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02!
z_log_var/MatMul/ReadVariableOp�
z_log_var/MatMulMatMulflatten_2/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/MatMul�
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
z_log_var/BiasAddY
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2	
z/Shapex
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
z/strided_slice/stack|
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
z/strided_slice/stack_1|
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
z/strided_slice/stack_2�
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
z/strided_slicet
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
z/random_normal/shape/1�
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2
z/random_normal/shapeq
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
z/random_normal/meanu
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
z/random_normal/stddev�
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2���2&
$z/random_normal/RandomStandardNormal�
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2
z/random_normal/mul�
z/random_normalAddz/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2
z/random_normalc
z/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
z/Expg
z/mulMul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:���������	2
z/mulm
z/addAddV2z_mean/BiasAdd:output:0	z/mul:z:0*
T0*'
_output_shapes
:���������	2
z/add�
IdentityIdentityz_mean/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity�

Identity_1Identityz_log_var/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity_1�

Identity_2Identity	z/add:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:����������::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_160932
encoder_input?
;vae_cnn_encoder_output_conv1_conv2d_readvariableop_resource@
<vae_cnn_encoder_output_conv1_biasadd_readvariableop_resource?
;vae_cnn_encoder_output_conv2_conv2d_readvariableop_resource@
<vae_cnn_encoder_output_conv2_biasadd_readvariableop_resource@
<vae_cnn_encoder_output_z_mean_matmul_readvariableop_resourceA
=vae_cnn_encoder_output_z_mean_biasadd_readvariableop_resourceC
?vae_cnn_encoder_output_z_log_var_matmul_readvariableop_resourceD
@vae_cnn_encoder_output_z_log_var_biasadd_readvariableop_resourceK
Gvae_cnn_decoder_output_deconv1_conv2d_transpose_readvariableop_resourceB
>vae_cnn_decoder_output_deconv1_biasadd_readvariableop_resourceK
Gvae_cnn_decoder_output_deconv2_conv2d_transpose_readvariableop_resourceB
>vae_cnn_decoder_output_deconv2_biasadd_readvariableop_resourceA
=vae_cnn_decoder_output_dense_1_matmul_readvariableop_resourceB
>vae_cnn_decoder_output_dense_1_biasadd_readvariableop_resource
identity��$vae_cnn/conv1/BiasAdd/ReadVariableOp�#vae_cnn/conv1/Conv2D/ReadVariableOp�$vae_cnn/conv2/BiasAdd/ReadVariableOp�#vae_cnn/conv2/Conv2D/ReadVariableOp�5vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp�>vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp�5vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp�>vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp�5vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp�4vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp�3vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp�2vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp�3vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp�2vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp�7vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp�6vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp�4vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp�3vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp�(vae_cnn/z_log_var/BiasAdd/ReadVariableOp�'vae_cnn/z_log_var/MatMul/ReadVariableOp�%vae_cnn/z_mean/BiasAdd/ReadVariableOp�$vae_cnn/z_mean/MatMul/ReadVariableOp�
&vae_cnn/encoder_output/reshape_2/ShapeShapeencoder_input*
T0*
_output_shapes
:2(
&vae_cnn/encoder_output/reshape_2/Shape�
4vae_cnn/encoder_output/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae_cnn/encoder_output/reshape_2/strided_slice/stack�
6vae_cnn/encoder_output/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/encoder_output/reshape_2/strided_slice/stack_1�
6vae_cnn/encoder_output/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/encoder_output/reshape_2/strided_slice/stack_2�
.vae_cnn/encoder_output/reshape_2/strided_sliceStridedSlice/vae_cnn/encoder_output/reshape_2/Shape:output:0=vae_cnn/encoder_output/reshape_2/strided_slice/stack:output:0?vae_cnn/encoder_output/reshape_2/strided_slice/stack_1:output:0?vae_cnn/encoder_output/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae_cnn/encoder_output/reshape_2/strided_slice�
0vae_cnn/encoder_output/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 22
0vae_cnn/encoder_output/reshape_2/Reshape/shape/1�
0vae_cnn/encoder_output/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 22
0vae_cnn/encoder_output/reshape_2/Reshape/shape/2�
0vae_cnn/encoder_output/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0vae_cnn/encoder_output/reshape_2/Reshape/shape/3�
.vae_cnn/encoder_output/reshape_2/Reshape/shapePack7vae_cnn/encoder_output/reshape_2/strided_slice:output:09vae_cnn/encoder_output/reshape_2/Reshape/shape/1:output:09vae_cnn/encoder_output/reshape_2/Reshape/shape/2:output:09vae_cnn/encoder_output/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.vae_cnn/encoder_output/reshape_2/Reshape/shape�
(vae_cnn/encoder_output/reshape_2/ReshapeReshapeencoder_input7vae_cnn/encoder_output/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2*
(vae_cnn/encoder_output/reshape_2/Reshape�
2vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOpReadVariableOp;vae_cnn_encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp�
#vae_cnn/encoder_output/conv1/Conv2DConv2D1vae_cnn/encoder_output/reshape_2/Reshape:output:0:vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2%
#vae_cnn/encoder_output/conv1/Conv2D�
3vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp�
$vae_cnn/encoder_output/conv1/BiasAddBiasAdd,vae_cnn/encoder_output/conv1/Conv2D:output:0;vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2&
$vae_cnn/encoder_output/conv1/BiasAdd�
!vae_cnn/encoder_output/conv1/ReluRelu-vae_cnn/encoder_output/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2#
!vae_cnn/encoder_output/conv1/Relu�
2vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOpReadVariableOp;vae_cnn_encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp�
#vae_cnn/encoder_output/conv2/Conv2DConv2D/vae_cnn/encoder_output/conv1/Relu:activations:0:vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2%
#vae_cnn/encoder_output/conv2/Conv2D�
3vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp�
$vae_cnn/encoder_output/conv2/BiasAddBiasAdd,vae_cnn/encoder_output/conv2/Conv2D:output:0;vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2&
$vae_cnn/encoder_output/conv2/BiasAdd�
!vae_cnn/encoder_output/conv2/ReluRelu-vae_cnn/encoder_output/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2#
!vae_cnn/encoder_output/conv2/Relu�
&vae_cnn/encoder_output/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2(
&vae_cnn/encoder_output/flatten_2/Const�
(vae_cnn/encoder_output/flatten_2/ReshapeReshape/vae_cnn/encoder_output/conv2/Relu:activations:0/vae_cnn/encoder_output/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2*
(vae_cnn/encoder_output/flatten_2/Reshape�
3vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype025
3vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp�
$vae_cnn/encoder_output/z_mean/MatMulMatMul1vae_cnn/encoder_output/flatten_2/Reshape:output:0;vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2&
$vae_cnn/encoder_output/z_mean/MatMul�
4vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOpReadVariableOp=vae_cnn_encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype026
4vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp�
%vae_cnn/encoder_output/z_mean/BiasAddBiasAdd.vae_cnn/encoder_output/z_mean/MatMul:product:0<vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2'
%vae_cnn/encoder_output/z_mean/BiasAdd�
6vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOpReadVariableOp?vae_cnn_encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype028
6vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp�
'vae_cnn/encoder_output/z_log_var/MatMulMatMul1vae_cnn/encoder_output/flatten_2/Reshape:output:0>vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2)
'vae_cnn/encoder_output/z_log_var/MatMul�
7vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOpReadVariableOp@vae_cnn_encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype029
7vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp�
(vae_cnn/encoder_output/z_log_var/BiasAddBiasAdd1vae_cnn/encoder_output/z_log_var/MatMul:product:0?vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2*
(vae_cnn/encoder_output/z_log_var/BiasAdd�
vae_cnn/encoder_output/z/ShapeShape.vae_cnn/encoder_output/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2 
vae_cnn/encoder_output/z/Shape�
,vae_cnn/encoder_output/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,vae_cnn/encoder_output/z/strided_slice/stack�
.vae_cnn/encoder_output/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.vae_cnn/encoder_output/z/strided_slice/stack_1�
.vae_cnn/encoder_output/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.vae_cnn/encoder_output/z/strided_slice/stack_2�
&vae_cnn/encoder_output/z/strided_sliceStridedSlice'vae_cnn/encoder_output/z/Shape:output:05vae_cnn/encoder_output/z/strided_slice/stack:output:07vae_cnn/encoder_output/z/strided_slice/stack_1:output:07vae_cnn/encoder_output/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&vae_cnn/encoder_output/z/strided_slice�
.vae_cnn/encoder_output/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	20
.vae_cnn/encoder_output/z/random_normal/shape/1�
,vae_cnn/encoder_output/z/random_normal/shapePack/vae_cnn/encoder_output/z/strided_slice:output:07vae_cnn/encoder_output/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,vae_cnn/encoder_output/z/random_normal/shape�
+vae_cnn/encoder_output/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+vae_cnn/encoder_output/z/random_normal/mean�
-vae_cnn/encoder_output/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-vae_cnn/encoder_output/z/random_normal/stddev�
;vae_cnn/encoder_output/z/random_normal/RandomStandardNormalRandomStandardNormal5vae_cnn/encoder_output/z/random_normal/shape:output:0*
T0*'
_output_shapes
:���������	*
dtype0*
seed���)*
seed2��h2=
;vae_cnn/encoder_output/z/random_normal/RandomStandardNormal�
*vae_cnn/encoder_output/z/random_normal/mulMulDvae_cnn/encoder_output/z/random_normal/RandomStandardNormal:output:06vae_cnn/encoder_output/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������	2,
*vae_cnn/encoder_output/z/random_normal/mul�
&vae_cnn/encoder_output/z/random_normalAdd.vae_cnn/encoder_output/z/random_normal/mul:z:04vae_cnn/encoder_output/z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������	2(
&vae_cnn/encoder_output/z/random_normal�
vae_cnn/encoder_output/z/ExpExp1vae_cnn/encoder_output/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
vae_cnn/encoder_output/z/Exp�
vae_cnn/encoder_output/z/mulMul vae_cnn/encoder_output/z/Exp:y:0*vae_cnn/encoder_output/z/random_normal:z:0*
T0*'
_output_shapes
:���������	2
vae_cnn/encoder_output/z/mul�
vae_cnn/encoder_output/z/addAddV2.vae_cnn/encoder_output/z_mean/BiasAdd:output:0 vae_cnn/encoder_output/z/mul:z:0*
T0*'
_output_shapes
:���������	2
vae_cnn/encoder_output/z/add�
&vae_cnn/decoder_output/reshape_3/ShapeShape vae_cnn/encoder_output/z/add:z:0*
T0*
_output_shapes
:2(
&vae_cnn/decoder_output/reshape_3/Shape�
4vae_cnn/decoder_output/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae_cnn/decoder_output/reshape_3/strided_slice/stack�
6vae_cnn/decoder_output/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/reshape_3/strided_slice/stack_1�
6vae_cnn/decoder_output/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/reshape_3/strided_slice/stack_2�
.vae_cnn/decoder_output/reshape_3/strided_sliceStridedSlice/vae_cnn/decoder_output/reshape_3/Shape:output:0=vae_cnn/decoder_output/reshape_3/strided_slice/stack:output:0?vae_cnn/decoder_output/reshape_3/strided_slice/stack_1:output:0?vae_cnn/decoder_output/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae_cnn/decoder_output/reshape_3/strided_slice�
0vae_cnn/decoder_output/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0vae_cnn/decoder_output/reshape_3/Reshape/shape/1�
0vae_cnn/decoder_output/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0vae_cnn/decoder_output/reshape_3/Reshape/shape/2�
0vae_cnn/decoder_output/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0vae_cnn/decoder_output/reshape_3/Reshape/shape/3�
.vae_cnn/decoder_output/reshape_3/Reshape/shapePack7vae_cnn/decoder_output/reshape_3/strided_slice:output:09vae_cnn/decoder_output/reshape_3/Reshape/shape/1:output:09vae_cnn/decoder_output/reshape_3/Reshape/shape/2:output:09vae_cnn/decoder_output/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.vae_cnn/decoder_output/reshape_3/Reshape/shape�
(vae_cnn/decoder_output/reshape_3/ReshapeReshape vae_cnn/encoder_output/z/add:z:07vae_cnn/decoder_output/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2*
(vae_cnn/decoder_output/reshape_3/Reshape�
$vae_cnn/decoder_output/deconv1/ShapeShape1vae_cnn/decoder_output/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2&
$vae_cnn/decoder_output/deconv1/Shape�
2vae_cnn/decoder_output/deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2vae_cnn/decoder_output/deconv1/strided_slice/stack�
4vae_cnn/decoder_output/deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4vae_cnn/decoder_output/deconv1/strided_slice/stack_1�
4vae_cnn/decoder_output/deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4vae_cnn/decoder_output/deconv1/strided_slice/stack_2�
,vae_cnn/decoder_output/deconv1/strided_sliceStridedSlice-vae_cnn/decoder_output/deconv1/Shape:output:0;vae_cnn/decoder_output/deconv1/strided_slice/stack:output:0=vae_cnn/decoder_output/deconv1/strided_slice/stack_1:output:0=vae_cnn/decoder_output/deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,vae_cnn/decoder_output/deconv1/strided_slice�
&vae_cnn/decoder_output/deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv1/stack/1�
&vae_cnn/decoder_output/deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv1/stack/2�
&vae_cnn/decoder_output/deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv1/stack/3�
$vae_cnn/decoder_output/deconv1/stackPack5vae_cnn/decoder_output/deconv1/strided_slice:output:0/vae_cnn/decoder_output/deconv1/stack/1:output:0/vae_cnn/decoder_output/deconv1/stack/2:output:0/vae_cnn/decoder_output/deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$vae_cnn/decoder_output/deconv1/stack�
4vae_cnn/decoder_output/deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae_cnn/decoder_output/deconv1/strided_slice_1/stack�
6vae_cnn/decoder_output/deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/deconv1/strided_slice_1/stack_1�
6vae_cnn/decoder_output/deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/deconv1/strided_slice_1/stack_2�
.vae_cnn/decoder_output/deconv1/strided_slice_1StridedSlice-vae_cnn/decoder_output/deconv1/stack:output:0=vae_cnn/decoder_output/deconv1/strided_slice_1/stack:output:0?vae_cnn/decoder_output/deconv1/strided_slice_1/stack_1:output:0?vae_cnn/decoder_output/deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae_cnn/decoder_output/deconv1/strided_slice_1�
>vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOpReadVariableOpGvae_cnn_decoder_output_deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02@
>vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp�
/vae_cnn/decoder_output/deconv1/conv2d_transposeConv2DBackpropInput-vae_cnn/decoder_output/deconv1/stack:output:0Fvae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp:value:01vae_cnn/decoder_output/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
21
/vae_cnn/decoder_output/deconv1/conv2d_transpose�
5vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOpReadVariableOp>vae_cnn_decoder_output_deconv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp�
&vae_cnn/decoder_output/deconv1/BiasAddBiasAdd8vae_cnn/decoder_output/deconv1/conv2d_transpose:output:0=vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2(
&vae_cnn/decoder_output/deconv1/BiasAdd�
#vae_cnn/decoder_output/deconv1/ReluRelu/vae_cnn/decoder_output/deconv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2%
#vae_cnn/decoder_output/deconv1/Relu�
$vae_cnn/decoder_output/deconv2/ShapeShape1vae_cnn/decoder_output/deconv1/Relu:activations:0*
T0*
_output_shapes
:2&
$vae_cnn/decoder_output/deconv2/Shape�
2vae_cnn/decoder_output/deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2vae_cnn/decoder_output/deconv2/strided_slice/stack�
4vae_cnn/decoder_output/deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4vae_cnn/decoder_output/deconv2/strided_slice/stack_1�
4vae_cnn/decoder_output/deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4vae_cnn/decoder_output/deconv2/strided_slice/stack_2�
,vae_cnn/decoder_output/deconv2/strided_sliceStridedSlice-vae_cnn/decoder_output/deconv2/Shape:output:0;vae_cnn/decoder_output/deconv2/strided_slice/stack:output:0=vae_cnn/decoder_output/deconv2/strided_slice/stack_1:output:0=vae_cnn/decoder_output/deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,vae_cnn/decoder_output/deconv2/strided_slice�
&vae_cnn/decoder_output/deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv2/stack/1�
&vae_cnn/decoder_output/deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv2/stack/2�
&vae_cnn/decoder_output/deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&vae_cnn/decoder_output/deconv2/stack/3�
$vae_cnn/decoder_output/deconv2/stackPack5vae_cnn/decoder_output/deconv2/strided_slice:output:0/vae_cnn/decoder_output/deconv2/stack/1:output:0/vae_cnn/decoder_output/deconv2/stack/2:output:0/vae_cnn/decoder_output/deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2&
$vae_cnn/decoder_output/deconv2/stack�
4vae_cnn/decoder_output/deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4vae_cnn/decoder_output/deconv2/strided_slice_1/stack�
6vae_cnn/decoder_output/deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/deconv2/strided_slice_1/stack_1�
6vae_cnn/decoder_output/deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6vae_cnn/decoder_output/deconv2/strided_slice_1/stack_2�
.vae_cnn/decoder_output/deconv2/strided_slice_1StridedSlice-vae_cnn/decoder_output/deconv2/stack:output:0=vae_cnn/decoder_output/deconv2/strided_slice_1/stack:output:0?vae_cnn/decoder_output/deconv2/strided_slice_1/stack_1:output:0?vae_cnn/decoder_output/deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.vae_cnn/decoder_output/deconv2/strided_slice_1�
>vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOpReadVariableOpGvae_cnn_decoder_output_deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02@
>vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp�
/vae_cnn/decoder_output/deconv2/conv2d_transposeConv2DBackpropInput-vae_cnn/decoder_output/deconv2/stack:output:0Fvae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp:value:01vae_cnn/decoder_output/deconv1/Relu:activations:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
21
/vae_cnn/decoder_output/deconv2/conv2d_transpose�
5vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOpReadVariableOp>vae_cnn_decoder_output_deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp�
&vae_cnn/decoder_output/deconv2/BiasAddBiasAdd8vae_cnn/decoder_output/deconv2/conv2d_transpose:output:0=vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2(
&vae_cnn/decoder_output/deconv2/BiasAdd�
#vae_cnn/decoder_output/deconv2/ReluRelu/vae_cnn/decoder_output/deconv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2%
#vae_cnn/decoder_output/deconv2/Relu�
&vae_cnn/decoder_output/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2(
&vae_cnn/decoder_output/flatten_3/Const�
(vae_cnn/decoder_output/flatten_3/ReshapeReshape1vae_cnn/decoder_output/deconv2/Relu:activations:0/vae_cnn/decoder_output/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2*
(vae_cnn/decoder_output/flatten_3/Reshape�
4vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOpReadVariableOp=vae_cnn_decoder_output_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype026
4vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp�
%vae_cnn/decoder_output/dense_1/MatMulMatMul1vae_cnn/decoder_output/flatten_3/Reshape:output:0<vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%vae_cnn/decoder_output/dense_1/MatMul�
5vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOpReadVariableOp>vae_cnn_decoder_output_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype027
5vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp�
&vae_cnn/decoder_output/dense_1/BiasAddBiasAdd/vae_cnn/decoder_output/dense_1/MatMul:product:0=vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&vae_cnn/decoder_output/dense_1/BiasAdd�
&vae_cnn/decoder_output/dense_1/SigmoidSigmoid/vae_cnn/decoder_output/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&vae_cnn/decoder_output/dense_1/Sigmoido
vae_cnn/reshape_2/ShapeShapeencoder_input*
T0*
_output_shapes
:2
vae_cnn/reshape_2/Shape�
%vae_cnn/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%vae_cnn/reshape_2/strided_slice/stack�
'vae_cnn/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'vae_cnn/reshape_2/strided_slice/stack_1�
'vae_cnn/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'vae_cnn/reshape_2/strided_slice/stack_2�
vae_cnn/reshape_2/strided_sliceStridedSlice vae_cnn/reshape_2/Shape:output:0.vae_cnn/reshape_2/strided_slice/stack:output:00vae_cnn/reshape_2/strided_slice/stack_1:output:00vae_cnn/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
vae_cnn/reshape_2/strided_slice�
!vae_cnn/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!vae_cnn/reshape_2/Reshape/shape/1�
!vae_cnn/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2#
!vae_cnn/reshape_2/Reshape/shape/2�
!vae_cnn/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!vae_cnn/reshape_2/Reshape/shape/3�
vae_cnn/reshape_2/Reshape/shapePack(vae_cnn/reshape_2/strided_slice:output:0*vae_cnn/reshape_2/Reshape/shape/1:output:0*vae_cnn/reshape_2/Reshape/shape/2:output:0*vae_cnn/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
vae_cnn/reshape_2/Reshape/shape�
vae_cnn/reshape_2/ReshapeReshapeencoder_input(vae_cnn/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:���������  2
vae_cnn/reshape_2/Reshape�
6vae_cnn/tf.math.squared_difference_1/SquaredDifferenceSquaredDifference*vae_cnn/decoder_output/dense_1/Sigmoid:y:0encoder_input*
T0*(
_output_shapes
:����������28
6vae_cnn/tf.math.squared_difference_1/SquaredDifference�
#vae_cnn/conv1/Conv2D/ReadVariableOpReadVariableOp;vae_cnn_encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#vae_cnn/conv1/Conv2D/ReadVariableOp�
vae_cnn/conv1/Conv2DConv2D"vae_cnn/reshape_2/Reshape:output:0+vae_cnn/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
vae_cnn/conv1/Conv2D�
$vae_cnn/conv1/BiasAdd/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$vae_cnn/conv1/BiasAdd/ReadVariableOp�
vae_cnn/conv1/BiasAddBiasAddvae_cnn/conv1/Conv2D:output:0,vae_cnn/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
vae_cnn/conv1/BiasAdd�
vae_cnn/conv1/ReluReluvae_cnn/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
vae_cnn/conv1/Relu�
4vae_cnn/tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4vae_cnn/tf.math.reduce_mean_2/Mean/reduction_indices�
"vae_cnn/tf.math.reduce_mean_2/MeanMean:vae_cnn/tf.math.squared_difference_1/SquaredDifference:z:0=vae_cnn/tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������2$
"vae_cnn/tf.math.reduce_mean_2/Mean�
#vae_cnn/conv2/Conv2D/ReadVariableOpReadVariableOp;vae_cnn_encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#vae_cnn/conv2/Conv2D/ReadVariableOp�
vae_cnn/conv2/Conv2DConv2D vae_cnn/conv1/Relu:activations:0+vae_cnn/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
vae_cnn/conv2/Conv2D�
$vae_cnn/conv2/BiasAdd/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$vae_cnn/conv2/BiasAdd/ReadVariableOp�
vae_cnn/conv2/BiasAddBiasAddvae_cnn/conv2/Conv2D:output:0,vae_cnn/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
vae_cnn/conv2/BiasAdd�
vae_cnn/conv2/ReluReluvae_cnn/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
vae_cnn/conv2/Relu�
 vae_cnn/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 vae_cnn/tf.math.multiply_2/Mul/y�
vae_cnn/tf.math.multiply_2/MulMul+vae_cnn/tf.math.reduce_mean_2/Mean:output:0)vae_cnn/tf.math.multiply_2/Mul/y:output:0*
T0*#
_output_shapes
:���������2 
vae_cnn/tf.math.multiply_2/Mul�
vae_cnn/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@=  2
vae_cnn/flatten_2/Const�
vae_cnn/flatten_2/ReshapeReshape vae_cnn/conv2/Relu:activations:0 vae_cnn/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������z2
vae_cnn/flatten_2/Reshape�
'vae_cnn/z_log_var/MatMul/ReadVariableOpReadVariableOp?vae_cnn_encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02)
'vae_cnn/z_log_var/MatMul/ReadVariableOp�
vae_cnn/z_log_var/MatMulMatMul"vae_cnn/flatten_2/Reshape:output:0/vae_cnn/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
vae_cnn/z_log_var/MatMul�
(vae_cnn/z_log_var/BiasAdd/ReadVariableOpReadVariableOp@vae_cnn_encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02*
(vae_cnn/z_log_var/BiasAdd/ReadVariableOp�
vae_cnn/z_log_var/BiasAddBiasAdd"vae_cnn/z_log_var/MatMul:product:00vae_cnn/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
vae_cnn/z_log_var/BiasAdd�
$vae_cnn/z_mean/MatMul/ReadVariableOpReadVariableOp<vae_cnn_encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02&
$vae_cnn/z_mean/MatMul/ReadVariableOp�
vae_cnn/z_mean/MatMulMatMul"vae_cnn/flatten_2/Reshape:output:0,vae_cnn/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
vae_cnn/z_mean/MatMul�
%vae_cnn/z_mean/BiasAdd/ReadVariableOpReadVariableOp=vae_cnn_encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02'
%vae_cnn/z_mean/BiasAdd/ReadVariableOp�
vae_cnn/z_mean/BiasAddBiasAddvae_cnn/z_mean/MatMul:product:0-vae_cnn/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
vae_cnn/z_mean/BiasAdd�
vae_cnn/tf.math.exp_1/ExpExp"vae_cnn/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
vae_cnn/tf.math.exp_1/Exp�
vae_cnn/tf.math.square_1/SquareSquarevae_cnn/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2!
vae_cnn/tf.math.square_1/Square�
$vae_cnn/tf.__operators__.add_2/AddV2AddV2vae_cnn/tf.math.exp_1/Exp:y:0#vae_cnn/tf.math.square_1/Square:y:0*
T0*'
_output_shapes
:���������	2&
$vae_cnn/tf.__operators__.add_2/AddV2�
vae_cnn/tf.math.subtract_2/SubSub(vae_cnn/tf.__operators__.add_2/AddV2:z:0"vae_cnn/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2 
vae_cnn/tf.math.subtract_2/Sub�
 vae_cnn/tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 vae_cnn/tf.math.subtract_3/Sub/y�
vae_cnn/tf.math.subtract_3/SubSub"vae_cnn/tf.math.subtract_2/Sub:z:0)vae_cnn/tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:���������	2 
vae_cnn/tf.math.subtract_3/Sub�
2vae_cnn/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2vae_cnn/tf.math.reduce_sum_1/Sum/reduction_indices�
 vae_cnn/tf.math.reduce_sum_1/SumSum"vae_cnn/tf.math.subtract_3/Sub:z:0;vae_cnn/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������2"
 vae_cnn/tf.math.reduce_sum_1/Sum�
 vae_cnn/tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2"
 vae_cnn/tf.math.multiply_3/Mul/y�
vae_cnn/tf.math.multiply_3/MulMul)vae_cnn/tf.math.reduce_sum_1/Sum:output:0)vae_cnn/tf.math.multiply_3/Mul/y:output:0*
T0*#
_output_shapes
:���������2 
vae_cnn/tf.math.multiply_3/Mul�
$vae_cnn/tf.__operators__.add_3/AddV2AddV2"vae_cnn/tf.math.multiply_2/Mul:z:0"vae_cnn/tf.math.multiply_3/Mul:z:0*
T0*#
_output_shapes
:���������2&
$vae_cnn/tf.__operators__.add_3/AddV2�
#vae_cnn/tf.math.reduce_mean_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#vae_cnn/tf.math.reduce_mean_3/Const�
"vae_cnn/tf.math.reduce_mean_3/MeanMean(vae_cnn/tf.__operators__.add_3/AddV2:z:0,vae_cnn/tf.math.reduce_mean_3/Const:output:0*
T0*
_output_shapes
: 2$
"vae_cnn/tf.math.reduce_mean_3/Mean�	
IdentityIdentity*vae_cnn/decoder_output/dense_1/Sigmoid:y:0%^vae_cnn/conv1/BiasAdd/ReadVariableOp$^vae_cnn/conv1/Conv2D/ReadVariableOp%^vae_cnn/conv2/BiasAdd/ReadVariableOp$^vae_cnn/conv2/Conv2D/ReadVariableOp6^vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp?^vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp6^vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp?^vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp6^vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp5^vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp4^vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp3^vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp4^vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp3^vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp8^vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp7^vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp5^vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp4^vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp)^vae_cnn/z_log_var/BiasAdd/ReadVariableOp(^vae_cnn/z_log_var/MatMul/ReadVariableOp&^vae_cnn/z_mean/BiasAdd/ReadVariableOp%^vae_cnn/z_mean/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::2L
$vae_cnn/conv1/BiasAdd/ReadVariableOp$vae_cnn/conv1/BiasAdd/ReadVariableOp2J
#vae_cnn/conv1/Conv2D/ReadVariableOp#vae_cnn/conv1/Conv2D/ReadVariableOp2L
$vae_cnn/conv2/BiasAdd/ReadVariableOp$vae_cnn/conv2/BiasAdd/ReadVariableOp2J
#vae_cnn/conv2/Conv2D/ReadVariableOp#vae_cnn/conv2/Conv2D/ReadVariableOp2n
5vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp5vae_cnn/decoder_output/deconv1/BiasAdd/ReadVariableOp2�
>vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp>vae_cnn/decoder_output/deconv1/conv2d_transpose/ReadVariableOp2n
5vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp5vae_cnn/decoder_output/deconv2/BiasAdd/ReadVariableOp2�
>vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp>vae_cnn/decoder_output/deconv2/conv2d_transpose/ReadVariableOp2n
5vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp5vae_cnn/decoder_output/dense_1/BiasAdd/ReadVariableOp2l
4vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp4vae_cnn/decoder_output/dense_1/MatMul/ReadVariableOp2j
3vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp3vae_cnn/encoder_output/conv1/BiasAdd/ReadVariableOp2h
2vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp2vae_cnn/encoder_output/conv1/Conv2D/ReadVariableOp2j
3vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp3vae_cnn/encoder_output/conv2/BiasAdd/ReadVariableOp2h
2vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp2vae_cnn/encoder_output/conv2/Conv2D/ReadVariableOp2r
7vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp7vae_cnn/encoder_output/z_log_var/BiasAdd/ReadVariableOp2p
6vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp6vae_cnn/encoder_output/z_log_var/MatMul/ReadVariableOp2l
4vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp4vae_cnn/encoder_output/z_mean/BiasAdd/ReadVariableOp2j
3vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp3vae_cnn/encoder_output/z_mean/MatMul/ReadVariableOp2T
(vae_cnn/z_log_var/BiasAdd/ReadVariableOp(vae_cnn/z_log_var/BiasAdd/ReadVariableOp2R
'vae_cnn/z_log_var/MatMul/ReadVariableOp'vae_cnn/z_log_var/MatMul/ReadVariableOp2N
%vae_cnn/z_mean/BiasAdd/ReadVariableOp%vae_cnn/z_mean/BiasAdd/ReadVariableOp2L
$vae_cnn/z_mean/MatMul/ReadVariableOp$vae_cnn/z_mean/MatMul/ReadVariableOp:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�
�
J__inference_decoder_output_layer_call_and_return_conditional_losses_161501

inputs
deconv1_161484
deconv1_161486
deconv2_161489
deconv2_161491
dense_1_161495
dense_1_161497
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_1613902
reshape_3/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0deconv1_161484deconv1_161486*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv1_layer_call_and_return_conditional_losses_1613132!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_161489deconv2_161491*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_deconv2_layer_call_and_return_conditional_losses_1613622!
deconv2/StatefulPartitionedCall�
flatten_3/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_1614202
flatten_3/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_1_161495dense_1_161497*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1614392!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
B__inference_z_mean_layer_call_and_return_conditional_losses_161036

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�'
�
C__inference_deconv2_layer_call_and_return_conditional_losses_161362

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_162959

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������	:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
(__inference_vae_cnn_layer_call_fn_161900
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:����������: *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_vae_cnn_layer_call_and_return_conditional_losses_1618682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_nameencoder_input
�	
�
E__inference_z_log_var_layer_call_and_return_conditional_losses_162854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������z
 
_user_specified_nameinputs
�'
�
C__inference_deconv1_layer_call_and_return_conditional_losses_161313

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
/__inference_decoder_output_layer_call_fn_162757

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_1615012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
encoder_input7
serving_default_encoder_input:0����������C
decoder_output1
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"��
_tf_keras_network��{"class_name": "Functional", "name": "vae_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "vae_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}, "name": "encoder_output", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_3", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "name": "decoder_output", "inbound_nodes": [[["encoder_output", 1, 2, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_1", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_1", "inbound_nodes": [["z_log_var", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_1", "inbound_nodes": [["z_mean", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.exp_1", 0, 0, {"y": ["tf.math.square_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.convert_to_tensor_1", "trainable": true, "dtype": "float32", "function": "convert_to_tensor"}, "name": "tf.convert_to_tensor_1", "inbound_nodes": [["decoder_output", 1, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["encoder_input", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_2", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["z_log_var", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.squared_difference_1", "trainable": true, "dtype": "float32", "function": "math.squared_difference"}, "name": "tf.math.squared_difference_1", "inbound_nodes": [["tf.convert_to_tensor_1", 0, 0, {"y": ["tf.cast_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"y": 1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_2", "inbound_nodes": [["tf.math.squared_difference_1", 0, 0, {"axis": -1, "keepdims": false}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.subtract_3", 0, 0, {"axis": -1, "keepdims": false}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.math.reduce_mean_2", 0, 0, {"y": 1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.math.reduce_sum_1", 0, 0, {"y": 0.001, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.math.multiply_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"axis": null, "keepdims": false}]]}, {"class_name": "AddLoss", "config": {"name": "add_loss_1", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss_1", "inbound_nodes": [[["tf.math.reduce_mean_3", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["decoder_output", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1024]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vae_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}, "name": "encoder_output", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_3", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "name": "decoder_output", "inbound_nodes": [[["encoder_output", 1, 2, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_1", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_1", "inbound_nodes": [["z_log_var", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square_1", "inbound_nodes": [["z_mean", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.exp_1", 0, 0, {"y": ["tf.math.square_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.convert_to_tensor_1", "trainable": true, "dtype": "float32", "function": "convert_to_tensor"}, "name": "tf.convert_to_tensor_1", "inbound_nodes": [["decoder_output", 1, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["encoder_input", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_2", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["z_log_var", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.squared_difference_1", "trainable": true, "dtype": "float32", "function": "math.squared_difference"}, "name": "tf.math.squared_difference_1", "inbound_nodes": [["tf.convert_to_tensor_1", 0, 0, {"y": ["tf.cast_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"y": 1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_2", "inbound_nodes": [["tf.math.squared_difference_1", 0, 0, {"axis": -1, "keepdims": false}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.subtract_3", 0, 0, {"axis": -1, "keepdims": false}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.math.reduce_mean_2", 0, 0, {"y": 1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.math.reduce_sum_1", 0, 0, {"y": 0.001, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.math.multiply_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}, "name": "tf.math.reduce_mean_3", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"axis": null, "keepdims": false}]]}, {"class_name": "AddLoss", "config": {"name": "add_loss_1", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss_1", "inbound_nodes": [[["tf.math.reduce_mean_3", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["decoder_output", 1, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
�L
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
	layer_with_weights-2
	layer-5
layer_with_weights-3
layer-6
 layer-7
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�I
_tf_keras_network�I{"class_name": "Functional", "name": "encoder_output", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1024]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape_2", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}}}
�3
%layer-0
&layer-1
'layer_with_weights-0
'layer-2
(layer_with_weights-1
(layer-3
)layer-4
*layer_with_weights-2
*layer-5
+	variables
,trainable_variables
-regularization_losses
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�0
_tf_keras_network�0{"class_name": "Functional", "name": "decoder_output", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_3", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_3", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}}
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}}
�	

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
�	

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 20]}}
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15680}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15680]}}
�

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15680}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15680]}}
�
O	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.exp_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp_1", "trainable": true, "dtype": "float32", "function": "math.exp"}}
�
P	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.square_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.square_1", "trainable": true, "dtype": "float32", "function": "math.square"}}
�
Q	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
�
R	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.convert_to_tensor_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.convert_to_tensor_1", "trainable": true, "dtype": "float32", "function": "convert_to_tensor"}}
�
S	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.cast_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}}
�
T	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.subtract_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
�
U	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.squared_difference_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.squared_difference_1", "trainable": true, "dtype": "float32", "function": "math.squared_difference"}}
�
V	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.subtract_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
�
W	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.reduce_mean_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_mean_2", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}}
�
X	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
�
Y	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
�
Z	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
�
[	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
�
\	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.reduce_mean_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_mean_3", "trainable": true, "dtype": "float32", "function": "math.reduce_mean"}}
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AddLoss", "name": "add_loss_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_loss_1", "trainable": true, "dtype": "float32", "unconditional": false}}
�
aiter

bbeta_1

cbeta_2
	ddecay
elearning_rate3m�4m�9m�:m�Cm�Dm�Im�Jm�fm�gm�hm�im�jm�km�3v�4v�9v�:v�Cv�Dv�Iv�Jv�fv�gv�hv�iv�jv�kv�"
	optimizer
 "
trackable_dict_wrapper
�
30
41
92
:3
I4
J5
C6
D7
f8
g9
h10
i11
j12
k13"
trackable_list_wrapper
�
30
41
92
:3
I4
J5
C6
D7
f8
g9
h10
i11
j12
k13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables
	variables
mmetrics
trainable_variables
regularization_losses
nlayer_metrics

olayers
player_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�

q	variables
rtrainable_variables
sregularization_losses
t	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Lambda", "name": "z", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
X
30
41
92
:3
I4
J5
C6
D7"
trackable_list_wrapper
X
30
41
92
:3
I4
J5
C6
D7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
unon_trainable_variables
!	variables
vmetrics
"trainable_variables
#regularization_losses
wlayer_metrics

xlayers
ylayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "z_sampling", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}}
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}}
�


fkernel
gbias
~	variables
trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "deconv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 1]}}
�


hkernel
ibias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "deconv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 20]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

jkernel
kbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 980}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 980]}}
J
f0
g1
h2
i3
j4
k5"
trackable_list_wrapper
J
f0
g1
h2
i3
j4
k5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
+	variables
�metrics
,trainable_variables
-regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
/	variables
0trainable_variables
1regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1/kernel
:2
conv1/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
5	variables
6trainable_variables
7regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
;	variables
<trainable_variables
=regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
?	variables
@trainable_variables
Aregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!	�z	2z_log_var/kernel
:	2z_log_var/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
E	variables
Ftrainable_variables
Gregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�z	2z_mean/kernel
:	2z_mean/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
K	variables
Ltrainable_variables
Mregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
]	variables
^trainable_variables
_regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(:&2deconv1/kernel
:2deconv1/bias
(:&2deconv2/kernel
:2deconv2/bias
": 
��2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
q	variables
rtrainable_variables
sregularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
	5
6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
z	variables
{trainable_variables
|regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
~	variables
trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�metrics
�	variables
�trainable_variables
�regularization_losses
�layer_metrics
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
+:)2Adam/conv1/kernel/m
:2Adam/conv1/bias/m
+:)2Adam/conv2/kernel/m
:2Adam/conv2/bias/m
(:&	�z	2Adam/z_log_var/kernel/m
!:	2Adam/z_log_var/bias/m
%:#	�z	2Adam/z_mean/kernel/m
:	2Adam/z_mean/bias/m
-:+2Adam/deconv1/kernel/m
:2Adam/deconv1/bias/m
-:+2Adam/deconv2/kernel/m
:2Adam/deconv2/bias/m
':%
��2Adam/dense_1/kernel/m
 :�2Adam/dense_1/bias/m
+:)2Adam/conv1/kernel/v
:2Adam/conv1/bias/v
+:)2Adam/conv2/kernel/v
:2Adam/conv2/bias/v
(:&	�z	2Adam/z_log_var/kernel/v
!:	2Adam/z_log_var/bias/v
%:#	�z	2Adam/z_mean/kernel/v
:	2Adam/z_mean/bias/v
-:+2Adam/deconv1/kernel/v
:2Adam/deconv1/bias/v
-:+2Adam/deconv2/kernel/v
:2Adam/deconv2/bias/v
':%
��2Adam/dense_1/kernel/v
 :�2Adam/dense_1/bias/v
�2�
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162378
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161723
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162213
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161794�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_vae_cnn_layer_call_fn_162412
(__inference_vae_cnn_layer_call_fn_161900
(__inference_vae_cnn_layer_call_fn_162005
(__inference_vae_cnn_layer_call_fn_162446�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_160932�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *-�*
(�%
encoder_input����������
�2�
J__inference_encoder_output_layer_call_and_return_conditional_losses_162505
J__inference_encoder_output_layer_call_and_return_conditional_losses_162564
J__inference_encoder_output_layer_call_and_return_conditional_losses_161165
J__inference_encoder_output_layer_call_and_return_conditional_losses_161136�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_encoder_output_layer_call_fn_162614
/__inference_encoder_output_layer_call_fn_161274
/__inference_encoder_output_layer_call_fn_161220
/__inference_encoder_output_layer_call_fn_162589�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_decoder_output_layer_call_and_return_conditional_losses_162740
J__inference_decoder_output_layer_call_and_return_conditional_losses_162677
J__inference_decoder_output_layer_call_and_return_conditional_losses_161477
J__inference_decoder_output_layer_call_and_return_conditional_losses_161456�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_decoder_output_layer_call_fn_161554
/__inference_decoder_output_layer_call_fn_161516
/__inference_decoder_output_layer_call_fn_162757
/__inference_decoder_output_layer_call_fn_162774�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_reshape_2_layer_call_and_return_conditional_losses_162788�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_reshape_2_layer_call_fn_162793�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv1_layer_call_and_return_conditional_losses_162804�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv1_layer_call_fn_162813�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv2_layer_call_and_return_conditional_losses_162824�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv2_layer_call_fn_162833�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_flatten_2_layer_call_and_return_conditional_losses_162839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_2_layer_call_fn_162844�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_z_log_var_layer_call_and_return_conditional_losses_162854�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_z_log_var_layer_call_fn_162863�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_z_mean_layer_call_and_return_conditional_losses_162873�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_z_mean_layer_call_fn_162882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_add_loss_1_layer_call_and_return_conditional_losses_162887�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_add_loss_1_layer_call_fn_162893�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_162048encoder_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
=__inference_z_layer_call_and_return_conditional_losses_162913
=__inference_z_layer_call_and_return_conditional_losses_162933�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference_z_layer_call_fn_162939
"__inference_z_layer_call_fn_162945�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_reshape_3_layer_call_and_return_conditional_losses_162959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_reshape_3_layer_call_fn_162964�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_deconv1_layer_call_and_return_conditional_losses_161313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_deconv1_layer_call_fn_161323�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
C__inference_deconv2_layer_call_and_return_conditional_losses_161362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_deconv2_layer_call_fn_161372�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
E__inference_flatten_3_layer_call_and_return_conditional_losses_162976�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_3_layer_call_fn_162981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_162992�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_163001�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_160932�349:IJCDfghijk7�4
-�*
(�%
encoder_input����������
� "@�=
;
decoder_output)�&
decoder_output�����������
F__inference_add_loss_1_layer_call_and_return_conditional_losses_162887D�
�
�
inputs 
� ""�

�
0 
�
�	
1/0 X
+__inference_add_loss_1_layer_call_fn_162893)�
�
�
inputs 
� "� �
A__inference_conv1_layer_call_and_return_conditional_losses_162804l347�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������
� �
&__inference_conv1_layer_call_fn_162813_347�4
-�*
(�%
inputs���������  
� " �����������
A__inference_conv2_layer_call_and_return_conditional_losses_162824l9:7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
&__inference_conv2_layer_call_fn_162833_9:7�4
-�*
(�%
inputs���������
� " �����������
J__inference_decoder_output_layer_call_and_return_conditional_losses_161456mfghijk;�8
1�.
$�!

z_sampling���������	
p

 
� "&�#
�
0����������
� �
J__inference_decoder_output_layer_call_and_return_conditional_losses_161477mfghijk;�8
1�.
$�!

z_sampling���������	
p 

 
� "&�#
�
0����������
� �
J__inference_decoder_output_layer_call_and_return_conditional_losses_162677ifghijk7�4
-�*
 �
inputs���������	
p

 
� "&�#
�
0����������
� �
J__inference_decoder_output_layer_call_and_return_conditional_losses_162740ifghijk7�4
-�*
 �
inputs���������	
p 

 
� "&�#
�
0����������
� �
/__inference_decoder_output_layer_call_fn_161516`fghijk;�8
1�.
$�!

z_sampling���������	
p

 
� "������������
/__inference_decoder_output_layer_call_fn_161554`fghijk;�8
1�.
$�!

z_sampling���������	
p 

 
� "������������
/__inference_decoder_output_layer_call_fn_162757\fghijk7�4
-�*
 �
inputs���������	
p

 
� "������������
/__inference_decoder_output_layer_call_fn_162774\fghijk7�4
-�*
 �
inputs���������	
p 

 
� "������������
C__inference_deconv1_layer_call_and_return_conditional_losses_161313�fgI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
(__inference_deconv1_layer_call_fn_161323�fgI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
C__inference_deconv2_layer_call_and_return_conditional_losses_161362�hiI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
(__inference_deconv2_layer_call_fn_161372�hiI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
C__inference_dense_1_layer_call_and_return_conditional_losses_162992fjk8�5
.�+
)�&
inputs������������������
� "&�#
�
0����������
� �
(__inference_dense_1_layer_call_fn_163001Yjk8�5
.�+
)�&
inputs������������������
� "������������
J__inference_encoder_output_layer_call_and_return_conditional_losses_161136�349:IJCD?�<
5�2
(�%
encoder_input����������
p

 
� "j�g
`�]
�
0/0���������	
�
0/1���������	
�
0/2���������	
� �
J__inference_encoder_output_layer_call_and_return_conditional_losses_161165�349:IJCD?�<
5�2
(�%
encoder_input����������
p 

 
� "j�g
`�]
�
0/0���������	
�
0/1���������	
�
0/2���������	
� �
J__inference_encoder_output_layer_call_and_return_conditional_losses_162505�349:IJCD8�5
.�+
!�
inputs����������
p

 
� "j�g
`�]
�
0/0���������	
�
0/1���������	
�
0/2���������	
� �
J__inference_encoder_output_layer_call_and_return_conditional_losses_162564�349:IJCD8�5
.�+
!�
inputs����������
p 

 
� "j�g
`�]
�
0/0���������	
�
0/1���������	
�
0/2���������	
� �
/__inference_encoder_output_layer_call_fn_161220�349:IJCD?�<
5�2
(�%
encoder_input����������
p

 
� "Z�W
�
0���������	
�
1���������	
�
2���������	�
/__inference_encoder_output_layer_call_fn_161274�349:IJCD?�<
5�2
(�%
encoder_input����������
p 

 
� "Z�W
�
0���������	
�
1���������	
�
2���������	�
/__inference_encoder_output_layer_call_fn_162589�349:IJCD8�5
.�+
!�
inputs����������
p

 
� "Z�W
�
0���������	
�
1���������	
�
2���������	�
/__inference_encoder_output_layer_call_fn_162614�349:IJCD8�5
.�+
!�
inputs����������
p 

 
� "Z�W
�
0���������	
�
1���������	
�
2���������	�
E__inference_flatten_2_layer_call_and_return_conditional_losses_162839a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������z
� �
*__inference_flatten_2_layer_call_fn_162844T7�4
-�*
(�%
inputs���������
� "�����������z�
E__inference_flatten_3_layer_call_and_return_conditional_losses_162976{I�F
?�<
:�7
inputs+���������������������������
� ".�+
$�!
0������������������
� �
*__inference_flatten_3_layer_call_fn_162981nI�F
?�<
:�7
inputs+���������������������������
� "!��������������������
E__inference_reshape_2_layer_call_and_return_conditional_losses_162788a0�-
&�#
!�
inputs����������
� "-�*
#� 
0���������  
� �
*__inference_reshape_2_layer_call_fn_162793T0�-
&�#
!�
inputs����������
� " ����������  �
E__inference_reshape_3_layer_call_and_return_conditional_losses_162959`/�,
%�"
 �
inputs���������	
� "-�*
#� 
0���������
� �
*__inference_reshape_3_layer_call_fn_162964S/�,
%�"
 �
inputs���������	
� " �����������
$__inference_signature_wrapper_162048�349:IJCDfghijkH�E
� 
>�;
9
encoder_input(�%
encoder_input����������"@�=
;
decoder_output)�&
decoder_output�����������
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161723�349:IJCDfghijk?�<
5�2
(�%
encoder_input����������
p

 
� "4�1
�
0����������
�
�	
1/0 �
C__inference_vae_cnn_layer_call_and_return_conditional_losses_161794�349:IJCDfghijk?�<
5�2
(�%
encoder_input����������
p 

 
� "4�1
�
0����������
�
�	
1/0 �
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162213�349:IJCDfghijk8�5
.�+
!�
inputs����������
p

 
� "4�1
�
0����������
�
�	
1/0 �
C__inference_vae_cnn_layer_call_and_return_conditional_losses_162378�349:IJCDfghijk8�5
.�+
!�
inputs����������
p 

 
� "4�1
�
0����������
�
�	
1/0 �
(__inference_vae_cnn_layer_call_fn_161900l349:IJCDfghijk?�<
5�2
(�%
encoder_input����������
p

 
� "������������
(__inference_vae_cnn_layer_call_fn_162005l349:IJCDfghijk?�<
5�2
(�%
encoder_input����������
p 

 
� "������������
(__inference_vae_cnn_layer_call_fn_162412e349:IJCDfghijk8�5
.�+
!�
inputs����������
p

 
� "������������
(__inference_vae_cnn_layer_call_fn_162446e349:IJCDfghijk8�5
.�+
!�
inputs����������
p 

 
� "������������
=__inference_z_layer_call_and_return_conditional_losses_162913�b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������	

 
p
� "%�"
�
0���������	
� �
=__inference_z_layer_call_and_return_conditional_losses_162933�b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������	

 
p 
� "%�"
�
0���������	
� �
"__inference_z_layer_call_fn_162939~b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������	

 
p
� "����������	�
"__inference_z_layer_call_fn_162945~b�_
X�U
K�H
"�
inputs/0���������	
"�
inputs/1���������	

 
p 
� "����������	�
E__inference_z_log_var_layer_call_and_return_conditional_losses_162854]CD0�-
&�#
!�
inputs����������z
� "%�"
�
0���������	
� ~
*__inference_z_log_var_layer_call_fn_162863PCD0�-
&�#
!�
inputs����������z
� "����������	�
B__inference_z_mean_layer_call_and_return_conditional_losses_162873]IJ0�-
&�#
!�
inputs����������z
� "%�"
�
0���������	
� {
'__inference_z_mean_layer_call_fn_162882PIJ0�-
&�#
!�
inputs����������z
� "����������	