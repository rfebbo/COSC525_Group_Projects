��
��
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
delete_old_dirsbool(�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8Ţ
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
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
 
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
�
&metrics
regularization_losses
'layer_metrics

(layers
)layer_regularization_losses
*non_trainable_variables
	variables
	trainable_variables
 
 
 
 
�
+layer_metrics
,metrics
regularization_losses

-layers
.layer_regularization_losses
/non_trainable_variables
	variables
trainable_variables
ZX
VARIABLE_VALUEdeconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdeconv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
0layer_metrics
1metrics
regularization_losses

2layers
3layer_regularization_losses
4non_trainable_variables
	variables
trainable_variables
ZX
VARIABLE_VALUEdeconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdeconv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
5layer_metrics
6metrics
regularization_losses

7layers
8layer_regularization_losses
9non_trainable_variables
	variables
trainable_variables
 
 
 
�
:layer_metrics
;metrics
regularization_losses

<layers
=layer_regularization_losses
>non_trainable_variables
	variables
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
�
?layer_metrics
@metrics
"regularization_losses

Alayers
Blayer_regularization_losses
Cnon_trainable_variables
#	variables
$trainable_variables
 
 
*
0
1
2
3
4
5
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
}
serving_default_z_samplingPlaceholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_z_samplingdeconv1/kerneldeconv1/biasdeconv2/kerneldeconv2/biasdense/kernel
dense/bias*
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
GPU2 *0J 8� *,
f'R%
#__inference_signature_wrapper_80244
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"deconv1/kernel/Read/ReadVariableOp deconv1/bias/Read/ReadVariableOp"deconv2/kernel/Read/ReadVariableOp deconv2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU2 *0J 8� *'
f"R 
__inference__traced_save_80501
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeconv1/kerneldeconv1/biasdeconv2/kerneldeconv2/biasdense/kernel
dense/bias*
Tin
	2*
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
GPU2 *0J 8� **
f%R#
!__inference__traced_restore_80529��
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_80435

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
�
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80210

inputs
deconv1_80193
deconv1_80195
deconv2_80198
deconv2_80200
dense_80204
dense_80206
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_800612
reshape_1/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0deconv1_80193deconv1_80195*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_799842!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_80198deconv2_80200*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_800332!
deconv2/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_800912
flatten_1/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_80204dense_80206*
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
GPU2 *0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_801102
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_decoder_output_layer_call_fn_80225

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
GPU2 *0J 8� *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_802102
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
�
�
__inference__traced_save_80501
file_prefix-
)savev2_deconv1_kernel_read_readvariableop+
'savev2_deconv1_bias_read_readvariableop-
)savev2_deconv2_kernel_read_readvariableop+
'savev2_deconv2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_deconv1_kernel_read_readvariableop'savev2_deconv1_bias_read_readvariableop)savev2_deconv2_kernel_read_readvariableop'savev2_deconv2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*Z
_input_shapesI
G: :::::
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: 
�
�
!__inference__traced_restore_80529
file_prefix#
assignvariableop_deconv1_kernel#
assignvariableop_1_deconv1_bias%
!assignvariableop_2_deconv2_kernel#
assignvariableop_3_deconv2_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_deconv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_deconv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_deconv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_deconv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80172

inputs
deconv1_80155
deconv1_80157
deconv2_80160
deconv2_80162
dense_80166
dense_80168
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_800612
reshape_1/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0deconv1_80155deconv1_80157*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_799842!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_80160deconv2_80162*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_800332!
deconv2/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_800912
flatten_1/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_80166dense_80168*
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
GPU2 *0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_801102
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
|
'__inference_deconv1_layer_call_fn_79994

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
GPU2 *0J 8� *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_799842
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
�N
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80307

inputs4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��deconv1/BiasAdd/ReadVariableOp�'deconv1/conv2d_transpose/ReadVariableOp�deconv2/BiasAdd/ReadVariableOp�'deconv2/conv2d_transpose/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOpX
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_1/Reshapeh
deconv1/ShapeShapereshape_1/Reshape:output:0*
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
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
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
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_1/Const�
flatten_1/ReshapeReshapedeconv2/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense/Sigmoid�
IdentityIdentitydense/Sigmoid:y:0^deconv1/BiasAdd/ReadVariableOp(^deconv1/conv2d_transpose/ReadVariableOp^deconv2/BiasAdd/ReadVariableOp(^deconv2/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
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
'deconv2/conv2d_transpose/ReadVariableOp'deconv2/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_decoder_output_layer_call_fn_80187

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
GPU2 *0J 8� *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_801722
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
�
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_80061

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
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_80418

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
�
�
.__inference_decoder_output_layer_call_fn_80387

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
GPU2 *0J 8� *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_801722
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
�
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80127

z_sampling
deconv1_80069
deconv1_80071
deconv2_80074
deconv2_80076
dense_80121
dense_80123
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall
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
GPU2 *0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_800612
reshape_1/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0deconv1_80069deconv1_80071*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_799842!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_80074deconv2_80076*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_800332!
deconv2/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_800912
flatten_1/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_80121dense_80123*
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
GPU2 *0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_801102
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�
�
#__inference_signature_wrapper_80244

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
GPU2 *0J 8� *)
f$R"
 __inference__wrapped_model_799452
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
�'
�
B__inference_deconv1_layer_call_and_return_conditional_losses_79984

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
.__inference_decoder_output_layer_call_fn_80404

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
GPU2 *0J 8� *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_802102
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
�N
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80370

inputs4
0deconv1_conv2d_transpose_readvariableop_resource+
'deconv1_biasadd_readvariableop_resource4
0deconv2_conv2d_transpose_readvariableop_resource+
'deconv2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��deconv1/BiasAdd/ReadVariableOp�'deconv1/conv2d_transpose/ReadVariableOp�deconv2/BiasAdd/ReadVariableOp�'deconv2/conv2d_transpose/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOpX
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_1/Reshapeh
deconv1/ShapeShapereshape_1/Reshape:output:0*
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
deconv1/conv2d_transposeConv2DBackpropInputdeconv1/stack:output:0/deconv1/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
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
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_1/Const�
flatten_1/ReshapeReshapedeconv2/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense/Sigmoid�
IdentityIdentitydense/Sigmoid:y:0^deconv1/BiasAdd/ReadVariableOp(^deconv1/conv2d_transpose/ReadVariableOp^deconv2/BiasAdd/ReadVariableOp(^deconv2/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
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
'deconv2/conv2d_transpose/ReadVariableOp'deconv2/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
|
'__inference_deconv2_layer_call_fn_80043

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
GPU2 *0J 8� *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_800332
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
�'
�
B__inference_deconv2_layer_call_and_return_conditional_losses_80033

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
�
E
)__inference_reshape_1_layer_call_fn_80423

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
GPU2 *0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_800612
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
�
z
%__inference_dense_layer_call_fn_80460

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
GPU2 *0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_801102
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
@__inference_dense_layer_call_and_return_conditional_losses_80451

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
�
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80148

z_sampling
deconv1_80131
deconv1_80133
deconv2_80136
deconv2_80138
dense_80142
dense_80144
identity��deconv1/StatefulPartitionedCall�deconv2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCall
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
GPU2 *0J 8� *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_800612
reshape_1/PartitionedCall�
deconv1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0deconv1_80131deconv1_80133*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv1_layer_call_and_return_conditional_losses_799842!
deconv1/StatefulPartitionedCall�
deconv2/StatefulPartitionedCallStatefulPartitionedCall(deconv1/StatefulPartitionedCall:output:0deconv2_80136deconv2_80138*
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
GPU2 *0J 8� *K
fFRD
B__inference_deconv2_layer_call_and_return_conditional_losses_800332!
deconv2/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall(deconv2/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_800912
flatten_1/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_80142dense_80144*
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
GPU2 *0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_801102
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0 ^deconv1/StatefulPartitionedCall ^deconv2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2B
deconv1/StatefulPartitionedCalldeconv1/StatefulPartitionedCall2B
deconv2/StatefulPartitionedCalldeconv2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_80091

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
�e
�
 __inference__wrapped_model_79945

z_samplingC
?decoder_output_deconv1_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv1_biasadd_readvariableop_resourceC
?decoder_output_deconv2_conv2d_transpose_readvariableop_resource:
6decoder_output_deconv2_biasadd_readvariableop_resource7
3decoder_output_dense_matmul_readvariableop_resource8
4decoder_output_dense_biasadd_readvariableop_resource
identity��-decoder_output/deconv1/BiasAdd/ReadVariableOp�6decoder_output/deconv1/conv2d_transpose/ReadVariableOp�-decoder_output/deconv2/BiasAdd/ReadVariableOp�6decoder_output/deconv2/conv2d_transpose/ReadVariableOp�+decoder_output/dense/BiasAdd/ReadVariableOp�*decoder_output/dense/MatMul/ReadVariableOpz
decoder_output/reshape_1/ShapeShape
z_sampling*
T0*
_output_shapes
:2 
decoder_output/reshape_1/Shape�
,decoder_output/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder_output/reshape_1/strided_slice/stack�
.decoder_output/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_1/strided_slice/stack_1�
.decoder_output/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder_output/reshape_1/strided_slice/stack_2�
&decoder_output/reshape_1/strided_sliceStridedSlice'decoder_output/reshape_1/Shape:output:05decoder_output/reshape_1/strided_slice/stack:output:07decoder_output/reshape_1/strided_slice/stack_1:output:07decoder_output/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder_output/reshape_1/strided_slice�
(decoder_output/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_1/Reshape/shape/1�
(decoder_output/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_1/Reshape/shape/2�
(decoder_output/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(decoder_output/reshape_1/Reshape/shape/3�
&decoder_output/reshape_1/Reshape/shapePack/decoder_output/reshape_1/strided_slice:output:01decoder_output/reshape_1/Reshape/shape/1:output:01decoder_output/reshape_1/Reshape/shape/2:output:01decoder_output/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&decoder_output/reshape_1/Reshape/shape�
 decoder_output/reshape_1/ReshapeReshape
z_sampling/decoder_output/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2"
 decoder_output/reshape_1/Reshape�
decoder_output/deconv1/ShapeShape)decoder_output/reshape_1/Reshape:output:0*
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
'decoder_output/deconv1/conv2d_transposeConv2DBackpropInput%decoder_output/deconv1/stack:output:0>decoder_output/deconv1/conv2d_transpose/ReadVariableOp:value:0)decoder_output/reshape_1/Reshape:output:0*
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
decoder_output/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2 
decoder_output/flatten_1/Const�
 decoder_output/flatten_1/ReshapeReshape)decoder_output/deconv2/Relu:activations:0'decoder_output/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2"
 decoder_output/flatten_1/Reshape�
*decoder_output/dense/MatMul/ReadVariableOpReadVariableOp3decoder_output_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02,
*decoder_output/dense/MatMul/ReadVariableOp�
decoder_output/dense/MatMulMatMul)decoder_output/flatten_1/Reshape:output:02decoder_output/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder_output/dense/MatMul�
+decoder_output/dense/BiasAdd/ReadVariableOpReadVariableOp4decoder_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+decoder_output/dense/BiasAdd/ReadVariableOp�
decoder_output/dense/BiasAddBiasAdd%decoder_output/dense/MatMul:product:03decoder_output/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
decoder_output/dense/BiasAdd�
decoder_output/dense/SigmoidSigmoid%decoder_output/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
decoder_output/dense/Sigmoid�
IdentityIdentity decoder_output/dense/Sigmoid:y:0.^decoder_output/deconv1/BiasAdd/ReadVariableOp7^decoder_output/deconv1/conv2d_transpose/ReadVariableOp.^decoder_output/deconv2/BiasAdd/ReadVariableOp7^decoder_output/deconv2/conv2d_transpose/ReadVariableOp,^decoder_output/dense/BiasAdd/ReadVariableOp+^decoder_output/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������	::::::2^
-decoder_output/deconv1/BiasAdd/ReadVariableOp-decoder_output/deconv1/BiasAdd/ReadVariableOp2p
6decoder_output/deconv1/conv2d_transpose/ReadVariableOp6decoder_output/deconv1/conv2d_transpose/ReadVariableOp2^
-decoder_output/deconv2/BiasAdd/ReadVariableOp-decoder_output/deconv2/BiasAdd/ReadVariableOp2p
6decoder_output/deconv2/conv2d_transpose/ReadVariableOp6decoder_output/deconv2/conv2d_transpose/ReadVariableOp2Z
+decoder_output/dense/BiasAdd/ReadVariableOp+decoder_output/dense/BiasAdd/ReadVariableOp2X
*decoder_output/dense/MatMul/ReadVariableOp*decoder_output/dense/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������	
$
_user_specified_name
z_sampling
�
E
)__inference_flatten_1_layer_call_fn_80440

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
GPU2 *0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_800912
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

�
@__inference_dense_layer_call_and_return_conditional_losses_80110

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
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

z_sampling3
serving_default_z_sampling:0���������	:
dense1
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�3
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*D&call_and_return_all_conditional_losses
E_default_save_signature
F__call__"�0
_tf_keras_network�0{"class_name": "Functional", "name": "decoder_output", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_1", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder_output", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}, "name": "z_sampling", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}, "name": "reshape_1", "inbound_nodes": [[["z_sampling", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv1", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "deconv2", "inbound_nodes": [[["deconv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["deconv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["z_sampling", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "z_sampling", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "z_sampling"}}
�
regularization_losses
	variables
trainable_variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 1]}}}
�


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "deconv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv1", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 1]}}
�


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "deconv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "deconv2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 20]}}
�
regularization_losses
	variables
trainable_variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
*O&call_and_return_all_conditional_losses
P__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1024, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 980}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 980]}}
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
�
&metrics
regularization_losses
'layer_metrics

(layers
)layer_regularization_losses
*non_trainable_variables
	variables
	trainable_variables
F__call__
E_default_save_signature
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
,
Qserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+layer_metrics
,metrics
regularization_losses

-layers
.layer_regularization_losses
/non_trainable_variables
	variables
trainable_variables
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
(:&2deconv1/kernel
:2deconv1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
0layer_metrics
1metrics
regularization_losses

2layers
3layer_regularization_losses
4non_trainable_variables
	variables
trainable_variables
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
(:&2deconv2/kernel
:2deconv2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
5layer_metrics
6metrics
regularization_losses

7layers
8layer_regularization_losses
9non_trainable_variables
	variables
trainable_variables
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
:layer_metrics
;metrics
regularization_losses

<layers
=layer_regularization_losses
>non_trainable_variables
	variables
trainable_variables
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
?layer_metrics
@metrics
"regularization_losses

Alayers
Blayer_regularization_losses
Cnon_trainable_variables
#	variables
$trainable_variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
�2�
I__inference_decoder_output_layer_call_and_return_conditional_losses_80370
I__inference_decoder_output_layer_call_and_return_conditional_losses_80127
I__inference_decoder_output_layer_call_and_return_conditional_losses_80307
I__inference_decoder_output_layer_call_and_return_conditional_losses_80148�
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
 __inference__wrapped_model_79945�
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
annotations� *)�&
$�!

z_sampling���������	
�2�
.__inference_decoder_output_layer_call_fn_80225
.__inference_decoder_output_layer_call_fn_80387
.__inference_decoder_output_layer_call_fn_80404
.__inference_decoder_output_layer_call_fn_80187�
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
D__inference_reshape_1_layer_call_and_return_conditional_losses_80418�
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
)__inference_reshape_1_layer_call_fn_80423�
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
B__inference_deconv1_layer_call_and_return_conditional_losses_79984�
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
'__inference_deconv1_layer_call_fn_79994�
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
B__inference_deconv2_layer_call_and_return_conditional_losses_80033�
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
'__inference_deconv2_layer_call_fn_80043�
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
D__inference_flatten_1_layer_call_and_return_conditional_losses_80435�
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
)__inference_flatten_1_layer_call_fn_80440�
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
@__inference_dense_layer_call_and_return_conditional_losses_80451�
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
%__inference_dense_layer_call_fn_80460�
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
#__inference_signature_wrapper_80244
z_sampling"�
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
 �
 __inference__wrapped_model_79945m !3�0
)�&
$�!

z_sampling���������	
� ".�+
)
dense �
dense�����������
I__inference_decoder_output_layer_call_and_return_conditional_losses_80127m !;�8
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_80148m !;�8
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_80307i !7�4
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_80370i !7�4
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
.__inference_decoder_output_layer_call_fn_80187` !;�8
1�.
$�!

z_sampling���������	
p

 
� "������������
.__inference_decoder_output_layer_call_fn_80225` !;�8
1�.
$�!

z_sampling���������	
p 

 
� "������������
.__inference_decoder_output_layer_call_fn_80387\ !7�4
-�*
 �
inputs���������	
p

 
� "������������
.__inference_decoder_output_layer_call_fn_80404\ !7�4
-�*
 �
inputs���������	
p 

 
� "������������
B__inference_deconv1_layer_call_and_return_conditional_losses_79984�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
'__inference_deconv1_layer_call_fn_79994�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
B__inference_deconv2_layer_call_and_return_conditional_losses_80033�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
'__inference_deconv2_layer_call_fn_80043�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
@__inference_dense_layer_call_and_return_conditional_losses_80451f !8�5
.�+
)�&
inputs������������������
� "&�#
�
0����������
� �
%__inference_dense_layer_call_fn_80460Y !8�5
.�+
)�&
inputs������������������
� "������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_80435{I�F
?�<
:�7
inputs+���������������������������
� ".�+
$�!
0������������������
� �
)__inference_flatten_1_layer_call_fn_80440nI�F
?�<
:�7
inputs+���������������������������
� "!��������������������
D__inference_reshape_1_layer_call_and_return_conditional_losses_80418`/�,
%�"
 �
inputs���������	
� "-�*
#� 
0���������
� �
)__inference_reshape_1_layer_call_fn_80423S/�,
%�"
 �
inputs���������	
� " �����������
#__inference_signature_wrapper_80244{ !A�>
� 
7�4
2

z_sampling$�!

z_sampling���������	".�+
)
dense �
dense����������