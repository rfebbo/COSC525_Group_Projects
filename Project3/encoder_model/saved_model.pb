Д╚
┌╛
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
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12unknown8ю│
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
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└z	*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	└z	*
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
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└z	*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	└z	*
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

NoOpNoOp
Ш
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╙
value╔B╞ B┐
▓
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
 
8
0
1
2
3
"4
#5
(6
)7
8
0
1
2
3
"4
#5
(6
)7
н
2metrics
	regularization_losses
3layer_metrics

4layers
5layer_regularization_losses
6non_trainable_variables

	variables
trainable_variables
 
 
 
 
н
7layer_metrics
8metrics
regularization_losses

9layers
:layer_regularization_losses
;non_trainable_variables
	variables
trainable_variables
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
<layer_metrics
=metrics
regularization_losses

>layers
?layer_regularization_losses
@non_trainable_variables
	variables
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Alayer_metrics
Bmetrics
regularization_losses

Clayers
Dlayer_regularization_losses
Enon_trainable_variables
	variables
trainable_variables
 
 
 
н
Flayer_metrics
Gmetrics
regularization_losses

Hlayers
Ilayer_regularization_losses
Jnon_trainable_variables
	variables
 trainable_variables
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
н
Klayer_metrics
Lmetrics
$regularization_losses

Mlayers
Nlayer_regularization_losses
Onon_trainable_variables
%	variables
&trainable_variables
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
н
Player_metrics
Qmetrics
*regularization_losses

Rlayers
Slayer_regularization_losses
Tnon_trainable_variables
+	variables
,trainable_variables
 
 
 
н
Ulayer_metrics
Vmetrics
.regularization_losses

Wlayers
Xlayer_regularization_losses
Ynon_trainable_variables
/	variables
0trainable_variables
 
 
8
0
1
2
3
4
5
6
7
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
 
В
serving_default_encoder_inputPlaceholder*(
_output_shapes
:         А*
dtype0*
shape:         А
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *,
f'R%
#__inference_signature_wrapper_79423
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╣
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU2 *0J 8В *'
f"R 
__inference__traced_save_79800
Ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2	*
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
GPU2 *0J 8В **
f%R#
!__inference__traced_restore_79834Ї√
╦

┘
@__inference_conv1_layer_call_and_return_conditional_losses_79621

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
∙
z
%__inference_conv2_layer_call_fn_79650

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_791182
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╦

┘
@__inference_conv2_layer_call_and_return_conditional_losses_79118

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ш
j
!__inference_z_layer_call_fn_79751
inputs_0
inputs_1
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         	
"
_user_specified_name
inputs/1
Х	
▌
D__inference_z_log_var_layer_call_and_return_conditional_losses_79184

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
╦

┘
@__inference_conv2_layer_call_and_return_conditional_losses_79641

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ї
k
<__inference_z_layer_call_and_return_conditional_losses_79739
inputs_0
inputs_1
identityИF
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
strided_slice/stack_2т
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
random_normal/shape/1Ш
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
 *  А?2
random_normal/stddev▄
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2ЇЙЗ2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:         	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         	2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:         	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         	
"
_user_specified_name
inputs/1
Т	
┌
A__inference_z_mean_layer_call_and_return_conditional_losses_79158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
з%
╗
I__inference_encoder_output_layer_call_and_return_conditional_losses_79319

inputs
conv1_79294
conv1_79296
conv2_79299
conv2_79301
z_mean_79305
z_mean_79307
z_log_var_79310
z_log_var_79312
identity

identity_1

identity_2Ивconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвz/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▌
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_790722
reshape/PartitionedCallй
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_79294conv1_79296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_790912
conv1/StatefulPartitionedCallп
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_79299conv2_79301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_791182
conv2/StatefulPartitionedCallЎ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_791402
flatten/PartitionedCallж
z_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_mean_79305z_mean_79307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_791582 
z_mean/StatefulPartitionedCall╡
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_79310z_log_var_79312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_791842#
!z_log_var/StatefulPartitionedCallй
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792202
z/StatefulPartitionedCallЬ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identityг

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Ы

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ш
j
!__inference_z_layer_call_fn_79745
inputs_0
inputs_1
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792202
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         	
"
_user_specified_name
inputs/1
∙
z
%__inference_conv1_layer_call_fn_79630

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_790912
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
ы
i
<__inference_z_layer_call_and_return_conditional_losses_79220

inputs
inputs_1
identityИD
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
strided_slice/stack_2т
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
random_normal/shape/1Ш
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
 *  А?2
random_normal/stddev▄
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2╘Эк2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:         	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         	2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:         	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_nameinputs
е
C
'__inference_flatten_layer_call_fn_79661

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_791402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └z2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╒I
ф
I__inference_encoder_output_layer_call_and_return_conditional_losses_79541

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

identity_2Ивconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpв z_log_var/BiasAdd/ReadVariableOpвz_log_var/MatMul/ReadVariableOpвz_mean/BiasAdd/ReadVariableOpвz_mean/MatMul/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeП
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:           2
reshape/Reshapeз
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp╚
conv1/Conv2DConv2Dreshape/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv1/Reluз
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @=  2
flatten/ConstТ
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         └z2
flatten/Reshapeг
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
z_mean/MatMul/ReadVariableOpЪ
z_mean/MatMulMatMulflatten/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_mean/MatMulб
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOpЭ
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_mean/BiasAddм
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02!
z_log_var/MatMul/ReadVariableOpг
z_log_var/MatMulMatMulflatten/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_log_var/MatMulк
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpй
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
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
z/strided_slice/stack_2ю
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
z/random_normal/shape/1а
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
 *  А?2
z/random_normal/stddevт
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2Ў╡Ц2&
$z/random_normal/RandomStandardNormal│
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
z/random_normal/mulУ
z/random_normalAddz/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
z/random_normalc
z/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
z/Expg
z/mulMul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:         	2
z/mulm
z/addAddV2z_mean/BiasAdd:output:0	z/mul:z:0*
T0*'
_output_shapes
:         	2
z/addщ
IdentityIdentityz_mean/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

IdentityЁ

Identity_1Identityz_log_var/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_1▀

Identity_2Identity	z/add:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2<
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
:         А
 
_user_specified_nameinputs
╝%
┬
I__inference_encoder_output_layer_call_and_return_conditional_losses_79258
encoder_input
conv1_79102
conv1_79104
conv2_79129
conv2_79131
z_mean_79169
z_mean_79171
z_log_var_79195
z_log_var_79197
identity

identity_1

identity_2Ивconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвz/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCallф
reshape/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_790722
reshape/PartitionedCallй
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_79102conv1_79104*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_790912
conv1/StatefulPartitionedCallп
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_79129conv2_79131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_791182
conv2/StatefulPartitionedCallЎ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_791402
flatten/PartitionedCallж
z_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_mean_79169z_mean_79171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_791582 
z_mean/StatefulPartitionedCall╡
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_79195z_log_var_79197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_791842#
!z_log_var/StatefulPartitionedCallй
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792202
z/StatefulPartitionedCallЬ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identityг

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Ы

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input
╟
∙
#__inference_signature_wrapper_79423
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

identity_2ИвStatefulPartitionedCall╥
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *)
f$R"
 __inference__wrapped_model_790542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input
ц
¤
.__inference_encoder_output_layer_call_fn_79566

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

identity_2ИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_encoder_output_layer_call_and_return_conditional_losses_793192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
х
^
B__inference_reshape_layer_call_and_return_conditional_losses_79605

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
strided_slice/stack_2т
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
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:           2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з%
╗
I__inference_encoder_output_layer_call_and_return_conditional_losses_79373

inputs
conv1_79348
conv1_79350
conv2_79353
conv2_79355
z_mean_79359
z_mean_79361
z_log_var_79364
z_log_var_79366
identity

identity_1

identity_2Ивconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвz/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▌
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_790722
reshape/PartitionedCallй
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_79348conv1_79350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_790912
conv1/StatefulPartitionedCallп
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_79353conv2_79355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_791182
conv2/StatefulPartitionedCallЎ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_791402
flatten/PartitionedCallж
z_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_mean_79359z_mean_79361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_791582 
z_mean/StatefulPartitionedCall╡
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_79364z_log_var_79366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_791842#
!z_log_var/StatefulPartitionedCallй
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792402
z/StatefulPartitionedCallЬ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identityг

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Ы

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Х	
▌
D__inference_z_log_var_layer_call_and_return_conditional_losses_79690

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
ц
¤
.__inference_encoder_output_layer_call_fn_79591

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

identity_2ИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_encoder_output_layer_call_and_return_conditional_losses_793732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
е
C
'__inference_reshape_layer_call_fn_79610

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_790722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▌
{
&__inference_z_mean_layer_call_fn_79680

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_791582
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
ї
k
<__inference_z_layer_call_and_return_conditional_losses_79719
inputs_0
inputs_1
identityИF
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
strided_slice/stack_2т
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
random_normal/shape/1Ш
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
 *  А?2
random_normal/stddev▄
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2и█В2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:         	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         	2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:         	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	:Q M
'
_output_shapes
:         	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         	
"
_user_specified_name
inputs/1
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_79140

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @=  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └z2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └z2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
х
^
B__inference_reshape_layer_call_and_return_conditional_losses_79072

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
strided_slice/stack_2т
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
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:           2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╦

┘
@__inference_conv1_layer_call_and_return_conditional_losses_79091

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╚%
Х
!__inference__traced_restore_79834
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias$
 assignvariableop_4_z_mean_kernel"
assignvariableop_5_z_mean_bias'
#assignvariableop_6_z_log_var_kernel%
!assignvariableop_7_z_log_var_bias

identity_9ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7▀
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ы
valueсB▐	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices╪
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1в
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2д
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3в
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp assignvariableop_4_z_mean_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_z_mean_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6и
AssignVariableOp_6AssignVariableOp#assignvariableop_6_z_log_var_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ж
AssignVariableOp_7AssignVariableOp!assignvariableop_7_z_log_var_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpО

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8А

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╜a
▓
 __inference__wrapped_model_79054
encoder_input7
3encoder_output_conv1_conv2d_readvariableop_resource8
4encoder_output_conv1_biasadd_readvariableop_resource7
3encoder_output_conv2_conv2d_readvariableop_resource8
4encoder_output_conv2_biasadd_readvariableop_resource8
4encoder_output_z_mean_matmul_readvariableop_resource9
5encoder_output_z_mean_biasadd_readvariableop_resource;
7encoder_output_z_log_var_matmul_readvariableop_resource<
8encoder_output_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ив+encoder_output/conv1/BiasAdd/ReadVariableOpв*encoder_output/conv1/Conv2D/ReadVariableOpв+encoder_output/conv2/BiasAdd/ReadVariableOpв*encoder_output/conv2/Conv2D/ReadVariableOpв/encoder_output/z_log_var/BiasAdd/ReadVariableOpв.encoder_output/z_log_var/MatMul/ReadVariableOpв,encoder_output/z_mean/BiasAdd/ReadVariableOpв+encoder_output/z_mean/MatMul/ReadVariableOpy
encoder_output/reshape/ShapeShapeencoder_input*
T0*
_output_shapes
:2
encoder_output/reshape/Shapeв
*encoder_output/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*encoder_output/reshape/strided_slice/stackж
,encoder_output/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,encoder_output/reshape/strided_slice/stack_1ж
,encoder_output/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,encoder_output/reshape/strided_slice/stack_2ь
$encoder_output/reshape/strided_sliceStridedSlice%encoder_output/reshape/Shape:output:03encoder_output/reshape/strided_slice/stack:output:05encoder_output/reshape/strided_slice/stack_1:output:05encoder_output/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$encoder_output/reshape/strided_sliceТ
&encoder_output/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder_output/reshape/Reshape/shape/1Т
&encoder_output/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder_output/reshape/Reshape/shape/2Т
&encoder_output/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&encoder_output/reshape/Reshape/shape/3─
$encoder_output/reshape/Reshape/shapePack-encoder_output/reshape/strided_slice:output:0/encoder_output/reshape/Reshape/shape/1:output:0/encoder_output/reshape/Reshape/shape/2:output:0/encoder_output/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$encoder_output/reshape/Reshape/shape├
encoder_output/reshape/ReshapeReshapeencoder_input-encoder_output/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:           2 
encoder_output/reshape/Reshape╘
*encoder_output/conv1/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv1/Conv2D/ReadVariableOpД
encoder_output/conv1/Conv2DConv2D'encoder_output/reshape/Reshape:output:02encoder_output/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
encoder_output/conv1/Conv2D╦
+encoder_output/conv1/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv1/BiasAdd/ReadVariableOp▄
encoder_output/conv1/BiasAddBiasAdd$encoder_output/conv1/Conv2D:output:03encoder_output/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
encoder_output/conv1/BiasAddЯ
encoder_output/conv1/ReluRelu%encoder_output/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2
encoder_output/conv1/Relu╘
*encoder_output/conv2/Conv2D/ReadVariableOpReadVariableOp3encoder_output_conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*encoder_output/conv2/Conv2D/ReadVariableOpД
encoder_output/conv2/Conv2DConv2D'encoder_output/conv1/Relu:activations:02encoder_output/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
encoder_output/conv2/Conv2D╦
+encoder_output/conv2/BiasAdd/ReadVariableOpReadVariableOp4encoder_output_conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+encoder_output/conv2/BiasAdd/ReadVariableOp▄
encoder_output/conv2/BiasAddBiasAdd$encoder_output/conv2/Conv2D:output:03encoder_output/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
encoder_output/conv2/BiasAddЯ
encoder_output/conv2/ReluRelu%encoder_output/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
encoder_output/conv2/ReluН
encoder_output/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @=  2
encoder_output/flatten/Const╬
encoder_output/flatten/ReshapeReshape'encoder_output/conv2/Relu:activations:0%encoder_output/flatten/Const:output:0*
T0*(
_output_shapes
:         └z2 
encoder_output/flatten/Reshape╨
+encoder_output/z_mean/MatMul/ReadVariableOpReadVariableOp4encoder_output_z_mean_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02-
+encoder_output/z_mean/MatMul/ReadVariableOp╓
encoder_output/z_mean/MatMulMatMul'encoder_output/flatten/Reshape:output:03encoder_output/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
encoder_output/z_mean/MatMul╬
,encoder_output/z_mean/BiasAdd/ReadVariableOpReadVariableOp5encoder_output_z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02.
,encoder_output/z_mean/BiasAdd/ReadVariableOp┘
encoder_output/z_mean/BiasAddBiasAdd&encoder_output/z_mean/MatMul:product:04encoder_output/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
encoder_output/z_mean/BiasAdd┘
.encoder_output/z_log_var/MatMul/ReadVariableOpReadVariableOp7encoder_output_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype020
.encoder_output/z_log_var/MatMul/ReadVariableOp▀
encoder_output/z_log_var/MatMulMatMul'encoder_output/flatten/Reshape:output:06encoder_output/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2!
encoder_output/z_log_var/MatMul╫
/encoder_output/z_log_var/BiasAdd/ReadVariableOpReadVariableOp8encoder_output_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype021
/encoder_output/z_log_var/BiasAdd/ReadVariableOpх
 encoder_output/z_log_var/BiasAddBiasAdd)encoder_output/z_log_var/MatMul:product:07encoder_output/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2"
 encoder_output/z_log_var/BiasAddЖ
encoder_output/z/ShapeShape&encoder_output/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder_output/z/ShapeЦ
$encoder_output/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$encoder_output/z/strided_slice/stackЪ
&encoder_output/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_1Ъ
&encoder_output/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder_output/z/strided_slice/stack_2╚
encoder_output/z/strided_sliceStridedSliceencoder_output/z/Shape:output:0-encoder_output/z/strided_slice/stack:output:0/encoder_output/z/strided_slice/stack_1:output:0/encoder_output/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
encoder_output/z/strided_sliceТ
&encoder_output/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2(
&encoder_output/z/random_normal/shape/1▄
$encoder_output/z/random_normal/shapePack'encoder_output/z/strided_slice:output:0/encoder_output/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$encoder_output/z/random_normal/shapeП
#encoder_output/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder_output/z/random_normal/meanУ
%encoder_output/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%encoder_output/z/random_normal/stddevП
3encoder_output/z/random_normal/RandomStandardNormalRandomStandardNormal-encoder_output/z/random_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2ос┘25
3encoder_output/z/random_normal/RandomStandardNormalя
"encoder_output/z/random_normal/mulMul<encoder_output/z/random_normal/RandomStandardNormal:output:0.encoder_output/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2$
"encoder_output/z/random_normal/mul╧
encoder_output/z/random_normalAdd&encoder_output/z/random_normal/mul:z:0,encoder_output/z/random_normal/mean:output:0*
T0*'
_output_shapes
:         	2 
encoder_output/z/random_normalР
encoder_output/z/ExpExp)encoder_output/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
encoder_output/z/Expг
encoder_output/z/mulMulencoder_output/z/Exp:y:0"encoder_output/z/random_normal:z:0*
T0*'
_output_shapes
:         	2
encoder_output/z/mulй
encoder_output/z/addAddV2&encoder_output/z_mean/BiasAdd:output:0encoder_output/z/mul:z:0*
T0*'
_output_shapes
:         	2
encoder_output/z/addт
IdentityIdentityencoder_output/z/add:z:0,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identityў

Identity_1Identity)encoder_output/z_log_var/BiasAdd:output:0,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_1Ї

Identity_2Identity&encoder_output/z_mean/BiasAdd:output:0,^encoder_output/conv1/BiasAdd/ReadVariableOp+^encoder_output/conv1/Conv2D/ReadVariableOp,^encoder_output/conv2/BiasAdd/ReadVariableOp+^encoder_output/conv2/Conv2D/ReadVariableOp0^encoder_output/z_log_var/BiasAdd/ReadVariableOp/^encoder_output/z_log_var/MatMul/ReadVariableOp-^encoder_output/z_mean/BiasAdd/ReadVariableOp,^encoder_output/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2Z
+encoder_output/conv1/BiasAdd/ReadVariableOp+encoder_output/conv1/BiasAdd/ReadVariableOp2X
*encoder_output/conv1/Conv2D/ReadVariableOp*encoder_output/conv1/Conv2D/ReadVariableOp2Z
+encoder_output/conv2/BiasAdd/ReadVariableOp+encoder_output/conv2/BiasAdd/ReadVariableOp2X
*encoder_output/conv2/Conv2D/ReadVariableOp*encoder_output/conv2/Conv2D/ReadVariableOp2b
/encoder_output/z_log_var/BiasAdd/ReadVariableOp/encoder_output/z_log_var/BiasAdd/ReadVariableOp2`
.encoder_output/z_log_var/MatMul/ReadVariableOp.encoder_output/z_log_var/MatMul/ReadVariableOp2\
,encoder_output/z_mean/BiasAdd/ReadVariableOp,encoder_output/z_mean/BiasAdd/ReadVariableOp2Z
+encoder_output/z_mean/MatMul/ReadVariableOp+encoder_output/z_mean/MatMul/ReadVariableOp:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input
у
~
)__inference_z_log_var_layer_call_fn_79699

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_791842
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
р
╒
__inference__traced_save_79800
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┘
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ы
valueсB▐	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЪ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesМ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*i
_input_shapesX
V: :::::	└z	:	:	└z	:	: 2(
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
:	└z	: 

_output_shapes
:	:%!

_output_shapes
:	└z	: 

_output_shapes
:	:	

_output_shapes
: 
╒I
ф
I__inference_encoder_output_layer_call_and_return_conditional_losses_79482

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

identity_2Ивconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpв z_log_var/BiasAdd/ReadVariableOpвz_log_var/MatMul/ReadVariableOpвz_mean/BiasAdd/ReadVariableOpвz_mean/MatMul/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeП
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*/
_output_shapes
:           2
reshape/Reshapeз
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp╚
conv1/Conv2DConv2Dreshape/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv1/Reluз
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @=  2
flatten/ConstТ
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         └z2
flatten/Reshapeг
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
z_mean/MatMul/ReadVariableOpЪ
z_mean/MatMulMatMulflatten/Reshape:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_mean/MatMulб
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
z_mean/BiasAdd/ReadVariableOpЭ
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_mean/BiasAddм
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02!
z_log_var/MatMul/ReadVariableOpг
z_log_var/MatMulMatMulflatten/Reshape:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
z_log_var/MatMulк
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpй
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
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
z/strided_slice/stack_2ю
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
z/random_normal/shape/1а
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
 *  А?2
z/random_normal/stddevт
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2╚й╣2&
$z/random_normal/RandomStandardNormal│
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
z/random_normal/mulУ
z/random_normalAddz/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
z/random_normalc
z/ExpExpz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         	2
z/Expg
z/mulMul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:         	2
z/mulm
z/addAddV2z_mean/BiasAdd:output:0	z/mul:z:0*
T0*'
_output_shapes
:         	2
z/addщ
IdentityIdentityz_mean/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

IdentityЁ

Identity_1Identityz_log_var/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_1▀

Identity_2Identity	z/add:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2<
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
:         А
 
_user_specified_nameinputs
Т	
┌
A__inference_z_mean_layer_call_and_return_conditional_losses_79671

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└z	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └z
 
_user_specified_nameinputs
ы
i
<__inference_z_layer_call_and_return_conditional_losses_79240

inputs
inputs_1
identityИD
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
strided_slice/stack_2т
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
random_normal/shape/1Ш
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
 *  А?2
random_normal/stddev▄
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         	*
dtype0*
seed▒ х)*
seed2ЄнУ2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         	2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         	2
random_normalM
ExpExpinputs_1*
T0*'
_output_shapes
:         	2
Exp_
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         	2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:         	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         	:         	:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs:OK
'
_output_shapes
:         	
 
_user_specified_nameinputs
╝%
┬
I__inference_encoder_output_layer_call_and_return_conditional_losses_79287
encoder_input
conv1_79262
conv1_79264
conv2_79267
conv2_79269
z_mean_79273
z_mean_79275
z_log_var_79278
z_log_var_79280
identity

identity_1

identity_2Ивconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвz/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCallф
reshape/PartitionedCallPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_790722
reshape/PartitionedCallй
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_79262conv1_79264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_790912
conv1/StatefulPartitionedCallп
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_79267conv2_79269*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_791182
conv2/StatefulPartitionedCallЎ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └z* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_791402
flatten/PartitionedCallж
z_mean/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_mean_79273z_mean_79275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_791582 
z_mean/StatefulPartitionedCall╡
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0z_log_var_79278z_log_var_79280*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_791842#
!z_log_var/StatefulPartitionedCallй
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_792402
z/StatefulPartitionedCallЬ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identityг

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Ы

Identity_2Identity"z/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_79656

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @=  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └z2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └z2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
√
Д
.__inference_encoder_output_layer_call_fn_79396
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

identity_2ИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_encoder_output_layer_call_and_return_conditional_losses_793732
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input
√
Д
.__inference_encoder_output_layer_call_fn_79342
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

identity_2ИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         	:         	:         	**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_encoder_output_layer_call_and_return_conditional_losses_793192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         А
'
_user_specified_nameencoder_input"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*м
serving_defaultШ
H
encoder_input7
serving_default_encoder_input:0         А5
z0
StatefulPartitionedCall:0         	=
	z_log_var0
StatefulPartitionedCall:1         	:
z_mean0
StatefulPartitionedCall:2         	tensorflow/serving/predict:╙·
┼L
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
*Z&call_and_return_all_conditional_losses
[_default_save_signature
\__call__"╣I
_tf_keras_networkЭI{"class_name": "Functional", "name": "encoder_output", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1024]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}, "name": "reshape", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "z", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["z", 0, 0]]}}}
√"°
_tf_keras_input_layer╪{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
ї
regularization_losses
	variables
trainable_variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"ц
_tf_keras_layer╠{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [32, 32, 1]}}}
ы	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"╞
_tf_keras_layerм{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
э	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"╚
_tf_keras_layerо{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 20]}}
т
regularization_losses
	variables
 trainable_variables
!	keras_api
*c&call_and_return_all_conditional_losses
d__call__"╙
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ї

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
*e&call_and_return_all_conditional_losses
f__call__"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15680}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15680]}}
√

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
*g&call_and_return_all_conditional_losses
h__call__"╓
_tf_keras_layer╝{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15680}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15680]}}
х

.regularization_losses
/	variables
0trainable_variables
1	keras_api
*i&call_and_return_all_conditional_losses
j__call__"╓	
_tf_keras_layer╝	{"class_name": "Lambda", "name": "z", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAYAAAAEAAAAQwAAAHNGAAAAfABcAn0BfQJ0AKABfAGhAWQBGQB9A3QAoAJ8\nAaEBZAIZAH0EdABqA3wDfARmAmQDjQF9BXwBdACgBHwCoQF8BRQAFwBTACkE+s5SZXBhcmFtZXRl\ncml6YXRpb24gdHJpY2sgYnkgc2FtcGxpbmcgZnJvbSBhbiBpc290cm9waWMgdW5pdCBHYXVzc2lh\nbi4KCiAgICAjIEFyZ3VtZW50cwogICAgICAgIGFyZ3MgKHRlbnNvcik6IG1lYW4gYW5kIGxvZyBv\nZiB2YXJpYW5jZSBvZiBRKHp8WCkKCiAgICAjIFJldHVybnMKICAgICAgICB6ICh0ZW5zb3IpOiBz\nYW1wbGVkIGxhdGVudCB2ZWN0b3IKICAgIOkAAAAA6QEAAAApAdoFc2hhcGUpBdoBS3IEAAAA2glp\nbnRfc2hhcGXaDXJhbmRvbV9ub3JtYWzaA2V4cCkG2gRhcmdz2gZ6X21lYW7aCXpfbG9nX3ZhctoF\nYmF0Y2jaA2RpbdoHZXBzaWxvbqkAcg8AAAD6WC9ob21lL3JmZWJiby9TY2hvb2wvQ2xhc3Nlcy9T\ncHJpbmdfMjAyMS9jb3NjNTI1L0NPU0M1MjVfR3JvdXBfUHJvamVjdHMvUHJvamVjdDMvdGFzazUu\ncHnaCHNhbXBsaW5nJQAAAHMKAAAAAAoIAg4BDgYQAg==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
X
0
1
2
3
"4
#5
(6
)7"
trackable_list_wrapper
X
0
1
2
3
"4
#5
(6
)7"
trackable_list_wrapper
╩
2metrics
	regularization_losses
3layer_metrics

4layers
5layer_regularization_losses
6non_trainable_variables

	variables
trainable_variables
\__call__
[_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
7layer_metrics
8metrics
regularization_losses

9layers
:layer_regularization_losses
;non_trainable_variables
	variables
trainable_variables
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1/kernel
:2
conv1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
<layer_metrics
=metrics
regularization_losses

>layers
?layer_regularization_losses
@non_trainable_variables
	variables
trainable_variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
Alayer_metrics
Bmetrics
regularization_losses

Clayers
Dlayer_regularization_losses
Enon_trainable_variables
	variables
trainable_variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Flayer_metrics
Gmetrics
regularization_losses

Hlayers
Ilayer_regularization_losses
Jnon_trainable_variables
	variables
 trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :	└z	2z_mean/kernel
:	2z_mean/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
н
Klayer_metrics
Lmetrics
$regularization_losses

Mlayers
Nlayer_regularization_losses
Onon_trainable_variables
%	variables
&trainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
#:!	└z	2z_log_var/kernel
:	2z_log_var/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
н
Player_metrics
Qmetrics
*regularization_losses

Rlayers
Slayer_regularization_losses
Tnon_trainable_variables
+	variables
,trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Ulayer_metrics
Vmetrics
.regularization_losses

Wlayers
Xlayer_regularization_losses
Ynon_trainable_variables
/	variables
0trainable_variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є2я
I__inference_encoder_output_layer_call_and_return_conditional_losses_79482
I__inference_encoder_output_layer_call_and_return_conditional_losses_79258
I__inference_encoder_output_layer_call_and_return_conditional_losses_79541
I__inference_encoder_output_layer_call_and_return_conditional_losses_79287└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
х2т
 __inference__wrapped_model_79054╜
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *-в*
(К%
encoder_input         А
Ж2Г
.__inference_encoder_output_layer_call_fn_79591
.__inference_encoder_output_layer_call_fn_79566
.__inference_encoder_output_layer_call_fn_79396
.__inference_encoder_output_layer_call_fn_79342└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ь2щ
B__inference_reshape_layer_call_and_return_conditional_losses_79605в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_reshape_layer_call_fn_79610в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_conv1_layer_call_and_return_conditional_losses_79621в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_conv1_layer_call_fn_79630в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_conv2_layer_call_and_return_conditional_losses_79641в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_conv2_layer_call_fn_79650в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_79656в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_flatten_layer_call_fn_79661в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_z_mean_layer_call_and_return_conditional_losses_79671в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_z_mean_layer_call_fn_79680в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_z_log_var_layer_call_and_return_conditional_losses_79690в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_z_log_var_layer_call_fn_79699в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┬2┐
<__inference_z_layer_call_and_return_conditional_losses_79719
<__inference_z_layer_call_and_return_conditional_losses_79739└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
М2Й
!__inference_z_layer_call_fn_79751
!__inference_z_layer_call_fn_79745└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨B═
#__inference_signature_wrapper_79423encoder_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 я
 __inference__wrapped_model_79054╩"#()7в4
-в*
(К%
encoder_input         А
к "ДкА
 
zК
z         	
0
	z_log_var#К 
	z_log_var         	
*
z_mean К
z_mean         	░
@__inference_conv1_layer_call_and_return_conditional_losses_79621l7в4
-в*
(К%
inputs           
к "-в*
#К 
0         
Ъ И
%__inference_conv1_layer_call_fn_79630_7в4
-в*
(К%
inputs           
к " К         ░
@__inference_conv2_layer_call_and_return_conditional_losses_79641l7в4
-в*
(К%
inputs         
к "-в*
#К 
0         
Ъ И
%__inference_conv2_layer_call_fn_79650_7в4
-в*
(К%
inputs         
к " К         Е
I__inference_encoder_output_layer_call_and_return_conditional_losses_79258╖"#()?в<
5в2
(К%
encoder_input         А
p

 
к "jвg
`Ъ]
К
0/0         	
К
0/1         	
К
0/2         	
Ъ Е
I__inference_encoder_output_layer_call_and_return_conditional_losses_79287╖"#()?в<
5в2
(К%
encoder_input         А
p 

 
к "jвg
`Ъ]
К
0/0         	
К
0/1         	
К
0/2         	
Ъ ■
I__inference_encoder_output_layer_call_and_return_conditional_losses_79482░"#()8в5
.в+
!К
inputs         А
p

 
к "jвg
`Ъ]
К
0/0         	
К
0/1         	
К
0/2         	
Ъ ■
I__inference_encoder_output_layer_call_and_return_conditional_losses_79541░"#()8в5
.в+
!К
inputs         А
p 

 
к "jвg
`Ъ]
К
0/0         	
К
0/1         	
К
0/2         	
Ъ ┌
.__inference_encoder_output_layer_call_fn_79342з"#()?в<
5в2
(К%
encoder_input         А
p

 
к "ZЪW
К
0         	
К
1         	
К
2         	┌
.__inference_encoder_output_layer_call_fn_79396з"#()?в<
5в2
(К%
encoder_input         А
p 

 
к "ZЪW
К
0         	
К
1         	
К
2         	╙
.__inference_encoder_output_layer_call_fn_79566а"#()8в5
.в+
!К
inputs         А
p

 
к "ZЪW
К
0         	
К
1         	
К
2         	╙
.__inference_encoder_output_layer_call_fn_79591а"#()8в5
.в+
!К
inputs         А
p 

 
к "ZЪW
К
0         	
К
1         	
К
2         	з
B__inference_flatten_layer_call_and_return_conditional_losses_79656a7в4
-в*
(К%
inputs         
к "&в#
К
0         └z
Ъ 
'__inference_flatten_layer_call_fn_79661T7в4
-в*
(К%
inputs         
к "К         └zз
B__inference_reshape_layer_call_and_return_conditional_losses_79605a0в-
&в#
!К
inputs         А
к "-в*
#К 
0           
Ъ 
'__inference_reshape_layer_call_fn_79610T0в-
&в#
!К
inputs         А
к " К           Г
#__inference_signature_wrapper_79423█"#()HвE
в 
>к;
9
encoder_input(К%
encoder_input         А"ДкА
 
zК
z         	
0
	z_log_var#К 
	z_log_var         	
*
z_mean К
z_mean         	╠
<__inference_z_layer_call_and_return_conditional_losses_79719Лbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         	

 
p
к "%в"
К
0         	
Ъ ╠
<__inference_z_layer_call_and_return_conditional_losses_79739Лbв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         	

 
p 
к "%в"
К
0         	
Ъ г
!__inference_z_layer_call_fn_79745~bв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         	

 
p
к "К         	г
!__inference_z_layer_call_fn_79751~bв_
XвU
KЪH
"К
inputs/0         	
"К
inputs/1         	

 
p 
к "К         	е
D__inference_z_log_var_layer_call_and_return_conditional_losses_79690]()0в-
&в#
!К
inputs         └z
к "%в"
К
0         	
Ъ }
)__inference_z_log_var_layer_call_fn_79699P()0в-
&в#
!К
inputs         └z
к "К         	в
A__inference_z_mean_layer_call_and_return_conditional_losses_79671]"#0в-
&в#
!К
inputs         └z
к "%в"
К
0         	
Ъ z
&__inference_z_mean_layer_call_fn_79680P"#0в-
&в#
!К
inputs         └z
к "К         	