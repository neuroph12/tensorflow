       БK"	  ђ;Ъ^оAbrain.Event:2Ћ▄бцН     фУe╚	#Ж;Ъ^оA"ЌФ
|
Input/PlaceholderPlaceholder*+
_output_shapes
:          * 
shape:          *
dtype0
u
Target/PlaceholderPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
^
conv1D_layers/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
Л
Dconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/shape*
T0*"
_output_shapes
:*

seed *6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
seed2 
ф
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d1/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d1/conv1d/kernel/AssignAssign#conv1D_layers/conv1d1/conv1d/kernel>conv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d1/conv1d/kernel/readIdentity#conv1D_layers/conv1d1/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d1/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d1/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d1/conv1d/bias/AssignAssign!conv1D_layers/conv1d1/conv1d/bias3conv1D_layers/conv1d1/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d1/conv1d/bias/readIdentity!conv1D_layers/conv1d1/conv1d/bias*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0
Ѓ
.conv1D_layers/conv1d1/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d1/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d1/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
М
3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims
ExpandDimsInput/Placeholder7conv1D_layers/conv1d1/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:          *
T0
{
9conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d1/conv1d/kernel/read9conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d1/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1*/
_output_shapes
:         *
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
╣
0conv1D_layers/conv1d1/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d1/conv1d/convolution/Conv2D*
squeeze_dims
*+
_output_shapes
:         *
T0
о
$conv1D_layers/conv1d1/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d1/conv1d/convolution/Squeeze&conv1D_layers/conv1d1/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d1/conv1d/ReluRelu$conv1D_layers/conv1d1/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d1/dropout/ShapeShape!conv1D_layers/conv1d1/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d1/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d1/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d1/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d1/dropout/random_uniform/subSub0conv1D_layers/conv1d1/dropout/random_uniform/max0conv1D_layers/conv1d1/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d1/dropout/random_uniform/mulMul:conv1D_layers/conv1d1/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d1/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d1/dropout/random_uniformAdd0conv1D_layers/conv1d1/dropout/random_uniform/mul0conv1D_layers/conv1d1/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d1/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d1/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d1/dropout/FloorFloor!conv1D_layers/conv1d1/dropout/add*
_output_shapes
:*
T0
Ї
!conv1D_layers/conv1d1/dropout/divRealDiv!conv1D_layers/conv1d1/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d1/dropout/mulMul!conv1D_layers/conv1d1/dropout/div#conv1D_layers/conv1d1/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d2/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d2/conv1d/kernel/AssignAssign#conv1D_layers/conv1d2/conv1d/kernel>conv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d2/conv1d/kernel/readIdentity#conv1D_layers/conv1d2/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d2/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d2/conv1d/bias
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
ј
(conv1D_layers/conv1d2/conv1d/bias/AssignAssign!conv1D_layers/conv1d2/conv1d/bias3conv1D_layers/conv1d2/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d2/conv1d/bias/readIdentity!conv1D_layers/conv1d2/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d2/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d2/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d2/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d1/dropout/mul7conv1D_layers/conv1d2/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d2/conv1d/kernel/read9conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d2/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1*/
_output_shapes
:         *
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
╣
0conv1D_layers/conv1d2/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d2/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d2/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d2/conv1d/convolution/Squeeze&conv1D_layers/conv1d2/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d2/conv1d/ReluRelu$conv1D_layers/conv1d2/conv1d/BiasAdd*
T0*+
_output_shapes
:         
ё
#conv1D_layers/conv1d2/dropout/ShapeShape!conv1D_layers/conv1d2/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d2/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d2/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d2/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d2/dropout/random_uniform/subSub0conv1D_layers/conv1d2/dropout/random_uniform/max0conv1D_layers/conv1d2/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d2/dropout/random_uniform/mulMul:conv1D_layers/conv1d2/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d2/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d2/dropout/random_uniformAdd0conv1D_layers/conv1d2/dropout/random_uniform/mul0conv1D_layers/conv1d2/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d2/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d2/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d2/dropout/FloorFloor!conv1D_layers/conv1d2/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d2/dropout/divRealDiv!conv1D_layers/conv1d2/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d2/dropout/mulMul!conv1D_layers/conv1d2/dropout/div#conv1D_layers/conv1d2/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
«
Lconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/shape*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0*
dtype0*
seed2 *

seed 
ф
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d3/conv1d/kernel
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Д
*conv1D_layers/conv1d3/conv1d/kernel/AssignAssign#conv1D_layers/conv1d3/conv1d/kernel>conv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d3/conv1d/kernel/readIdentity#conv1D_layers/conv1d3/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d3/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d3/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d3/conv1d/bias/AssignAssign!conv1D_layers/conv1d3/conv1d/bias3conv1D_layers/conv1d3/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d3/conv1d/bias/readIdentity!conv1D_layers/conv1d3/conv1d/bias*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
T0
Ѓ
.conv1D_layers/conv1d3/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d3/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d3/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d2/dropout/mul7conv1D_layers/conv1d3/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d3/conv1d/kernel/read9conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d3/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d3/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d3/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d3/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d3/conv1d/convolution/Squeeze&conv1D_layers/conv1d3/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d3/conv1d/ReluRelu$conv1D_layers/conv1d3/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d3/dropout/ShapeShape!conv1D_layers/conv1d3/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d3/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
u
0conv1D_layers/conv1d3/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d3/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d3/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d3/dropout/random_uniform/subSub0conv1D_layers/conv1d3/dropout/random_uniform/max0conv1D_layers/conv1d3/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d3/dropout/random_uniform/mulMul:conv1D_layers/conv1d3/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d3/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d3/dropout/random_uniformAdd0conv1D_layers/conv1d3/dropout/random_uniform/mul0conv1D_layers/conv1d3/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d3/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d3/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d3/dropout/FloorFloor!conv1D_layers/conv1d3/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d3/dropout/divRealDiv!conv1D_layers/conv1d3/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d3/dropout/mulMul!conv1D_layers/conv1d3/dropout/div#conv1D_layers/conv1d3/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d4/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d4/conv1d/kernel/AssignAssign#conv1D_layers/conv1d4/conv1d/kernel>conv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
validate_shape(*"
_output_shapes
:
Й
(conv1D_layers/conv1d4/conv1d/kernel/readIdentity#conv1D_layers/conv1d4/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d4/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d4/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d4/conv1d/bias/AssignAssign!conv1D_layers/conv1d4/conv1d/bias3conv1D_layers/conv1d4/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d4/conv1d/bias/readIdentity!conv1D_layers/conv1d4/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d4/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d4/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d4/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
с
3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d3/dropout/mul7conv1D_layers/conv1d4/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d4/conv1d/kernel/read9conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d4/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d4/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d4/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d4/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d4/conv1d/convolution/Squeeze&conv1D_layers/conv1d4/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d4/conv1d/ReluRelu$conv1D_layers/conv1d4/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d4/dropout/ShapeShape!conv1D_layers/conv1d4/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d4/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d4/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d4/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d4/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d4/dropout/random_uniform/subSub0conv1D_layers/conv1d4/dropout/random_uniform/max0conv1D_layers/conv1d4/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d4/dropout/random_uniform/mulMul:conv1D_layers/conv1d4/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d4/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d4/dropout/random_uniformAdd0conv1D_layers/conv1d4/dropout/random_uniform/mul0conv1D_layers/conv1d4/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d4/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d4/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d4/dropout/FloorFloor!conv1D_layers/conv1d4/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d4/dropout/divRealDiv!conv1D_layers/conv1d4/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d4/dropout/mulMul!conv1D_layers/conv1d4/dropout/div#conv1D_layers/conv1d4/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/shape*
T0*"
_output_shapes
:*

seed *6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
dtype0*
seed2 
ф
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/sub*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
▓
>conv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d5/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d5/conv1d/kernel/AssignAssign#conv1D_layers/conv1d5/conv1d/kernel>conv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d5/conv1d/kernel/readIdentity#conv1D_layers/conv1d5/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d5/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d5/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d5/conv1d/bias/AssignAssign!conv1D_layers/conv1d5/conv1d/bias3conv1D_layers/conv1d5/conv1d/bias/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
░
&conv1D_layers/conv1d5/conv1d/bias/readIdentity!conv1D_layers/conv1d5/conv1d/bias*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0
Ѓ
.conv1D_layers/conv1d5/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d5/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d5/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d4/dropout/mul7conv1D_layers/conv1d5/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d5/conv1d/kernel/read9conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d5/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d5/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d5/conv1d/convolution/Conv2D*
squeeze_dims
*+
_output_shapes
:         *
T0
о
$conv1D_layers/conv1d5/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d5/conv1d/convolution/Squeeze&conv1D_layers/conv1d5/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d5/conv1d/ReluRelu$conv1D_layers/conv1d5/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d5/dropout/ShapeShape!conv1D_layers/conv1d5/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d5/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
u
0conv1D_layers/conv1d5/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d5/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d5/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d5/dropout/random_uniform/subSub0conv1D_layers/conv1d5/dropout/random_uniform/max0conv1D_layers/conv1d5/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d5/dropout/random_uniform/mulMul:conv1D_layers/conv1d5/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d5/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d5/dropout/random_uniformAdd0conv1D_layers/conv1d5/dropout/random_uniform/mul0conv1D_layers/conv1d5/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d5/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d5/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d5/dropout/FloorFloor!conv1D_layers/conv1d5/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d5/dropout/divRealDiv!conv1D_layers/conv1d5/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d5/dropout/mulMul!conv1D_layers/conv1d5/dropout/div#conv1D_layers/conv1d5/dropout/Floor*
T0*+
_output_shapes
:         
Л
Dconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/shape*
T0*"
_output_shapes
:*

seed *6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
dtype0*
seed2 
ф
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/sub*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0
▓
>conv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d6/conv1d/kernel
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Д
*conv1D_layers/conv1d6/conv1d/kernel/AssignAssign#conv1D_layers/conv1d6/conv1d/kernel>conv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
validate_shape(*"
_output_shapes
:
Й
(conv1D_layers/conv1d6/conv1d/kernel/readIdentity#conv1D_layers/conv1d6/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d6/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d6/conv1d/bias
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
ј
(conv1D_layers/conv1d6/conv1d/bias/AssignAssign!conv1D_layers/conv1d6/conv1d/bias3conv1D_layers/conv1d6/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d6/conv1d/bias/readIdentity!conv1D_layers/conv1d6/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d6/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d6/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d6/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
с
3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d5/dropout/mul7conv1D_layers/conv1d6/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d6/conv1d/kernel/read9conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d6/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:         *
use_cudnn_on_gpu(
╣
0conv1D_layers/conv1d6/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d6/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d6/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d6/conv1d/convolution/Squeeze&conv1D_layers/conv1d6/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d6/conv1d/ReluRelu$conv1D_layers/conv1d6/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d6/dropout/ShapeShape!conv1D_layers/conv1d6/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d6/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
u
0conv1D_layers/conv1d6/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d6/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d6/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d6/dropout/random_uniform/subSub0conv1D_layers/conv1d6/dropout/random_uniform/max0conv1D_layers/conv1d6/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d6/dropout/random_uniform/mulMul:conv1D_layers/conv1d6/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d6/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d6/dropout/random_uniformAdd0conv1D_layers/conv1d6/dropout/random_uniform/mul0conv1D_layers/conv1d6/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d6/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d6/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d6/dropout/FloorFloor!conv1D_layers/conv1d6/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d6/dropout/divRealDiv!conv1D_layers/conv1d6/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d6/dropout/mulMul!conv1D_layers/conv1d6/dropout/div#conv1D_layers/conv1d6/dropout/Floor*
T0*+
_output_shapes
:         
Л
Dconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
ф
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d7/conv1d/kernel
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Д
*conv1D_layers/conv1d7/conv1d/kernel/AssignAssign#conv1D_layers/conv1d7/conv1d/kernel>conv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
validate_shape(*"
_output_shapes
:
Й
(conv1D_layers/conv1d7/conv1d/kernel/readIdentity#conv1D_layers/conv1d7/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d7/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d7/conv1d/bias
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
dtype0*
	container 
ј
(conv1D_layers/conv1d7/conv1d/bias/AssignAssign!conv1D_layers/conv1d7/conv1d/bias3conv1D_layers/conv1d7/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d7/conv1d/bias/readIdentity!conv1D_layers/conv1d7/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d7/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d7/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d7/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
с
3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d6/dropout/mul7conv1D_layers/conv1d7/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d7/conv1d/kernel/read9conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d7/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d7/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d7/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d7/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d7/conv1d/convolution/Squeeze&conv1D_layers/conv1d7/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d7/conv1d/ReluRelu$conv1D_layers/conv1d7/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d7/dropout/ShapeShape!conv1D_layers/conv1d7/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d7/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d7/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d7/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d7/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d7/dropout/random_uniform/subSub0conv1D_layers/conv1d7/dropout/random_uniform/max0conv1D_layers/conv1d7/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d7/dropout/random_uniform/mulMul:conv1D_layers/conv1d7/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d7/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d7/dropout/random_uniformAdd0conv1D_layers/conv1d7/dropout/random_uniform/mul0conv1D_layers/conv1d7/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d7/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d7/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d7/dropout/FloorFloor!conv1D_layers/conv1d7/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d7/dropout/divRealDiv!conv1D_layers/conv1d7/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d7/dropout/mulMul!conv1D_layers/conv1d7/dropout/div#conv1D_layers/conv1d7/dropout/Floor*
T0*+
_output_shapes
:         
Л
Dconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d8/conv1d/kernel
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Д
*conv1D_layers/conv1d8/conv1d/kernel/AssignAssign#conv1D_layers/conv1d8/conv1d/kernel>conv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d8/conv1d/kernel/readIdentity#conv1D_layers/conv1d8/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d8/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d8/conv1d/bias
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
ј
(conv1D_layers/conv1d8/conv1d/bias/AssignAssign!conv1D_layers/conv1d8/conv1d/bias3conv1D_layers/conv1d8/conv1d/bias/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
░
&conv1D_layers/conv1d8/conv1d/bias/readIdentity!conv1D_layers/conv1d8/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d8/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d8/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d8/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d7/dropout/mul7conv1D_layers/conv1d8/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d8/conv1d/kernel/read9conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d8/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1*/
_output_shapes
:         *
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingVALID
╣
0conv1D_layers/conv1d8/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d8/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d8/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d8/conv1d/convolution/Squeeze&conv1D_layers/conv1d8/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d8/conv1d/ReluRelu$conv1D_layers/conv1d8/conv1d/BiasAdd*
T0*+
_output_shapes
:         
ё
#conv1D_layers/conv1d8/dropout/ShapeShape!conv1D_layers/conv1d8/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d8/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d8/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d8/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d8/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d8/dropout/random_uniform/subSub0conv1D_layers/conv1d8/dropout/random_uniform/max0conv1D_layers/conv1d8/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d8/dropout/random_uniform/mulMul:conv1D_layers/conv1d8/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d8/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d8/dropout/random_uniformAdd0conv1D_layers/conv1d8/dropout/random_uniform/mul0conv1D_layers/conv1d8/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d8/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d8/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d8/dropout/FloorFloor!conv1D_layers/conv1d8/dropout/add*
_output_shapes
:*
T0
Ї
!conv1D_layers/conv1d8/dropout/divRealDiv!conv1D_layers/conv1d8/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d8/dropout/mulMul!conv1D_layers/conv1d8/dropout/div#conv1D_layers/conv1d8/dropout/Floor*+
_output_shapes
:         *
T0
t
2conv1D_layers/conv1d9/max_pooling1d/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
┘
.conv1D_layers/conv1d9/max_pooling1d/ExpandDims
ExpandDims!conv1D_layers/conv1d8/dropout/mul2conv1D_layers/conv1d9/max_pooling1d/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
з
+conv1D_layers/conv1d9/max_pooling1d/MaxPoolMaxPool.conv1D_layers/conv1d9/max_pooling1d/ExpandDims*
ksize
*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
░
+conv1D_layers/conv1d9/max_pooling1d/SqueezeSqueeze+conv1D_layers/conv1d9/max_pooling1d/MaxPool*
squeeze_dims
*
T0*+
_output_shapes
:         
x
Flatten/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/Squeeze*
T0*
out_type0*
_output_shapes
:
]
Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
\
Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ђ
Flatten/SliceSliceFlatten/ShapeFlatten/Slice/beginFlatten/Slice/size*
_output_shapes
:*
T0*
Index0
_
Flatten/Slice_1/beginConst*
valueB:*
_output_shapes
:*
dtype0
^
Flatten/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
є
Flatten/Slice_1SliceFlatten/ShapeFlatten/Slice_1/beginFlatten/Slice_1/size*
_output_shapes
:*
T0*
Index0
W
Flatten/ConstConst*
valueB: *
_output_shapes
:*
dtype0
r
Flatten/ProdProdFlatten/Slice_1Flatten/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
Flatten/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
w
Flatten/ExpandDims
ExpandDimsFlatten/ProdFlatten/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
U
Flatten/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ї
Flatten/concatConcatV2Flatten/SliceFlatten/ExpandDimsFlatten/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
Ќ
Flatten/ReshapeReshape+conv1D_layers/conv1d9/max_pooling1d/SqueezeFlatten/concat*
Tshape0*'
_output_shapes
:          *
T0
f
!classification_layers/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
█
Lclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shapeConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB"    
   *
_output_shapes
:*
dtype0
╬
Kclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/meanConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
л
Mclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddevConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
─
Vclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 

▀
Jclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulMulVclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalMclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 

═
Fclassification_layers/dense0/dense/kernel/Initializer/truncated_normalAddJclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulKclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mean*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
█
)classification_layers/dense0/dense/kernel
VariableV2*
shared_name *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
	container *
shape
: 
*
dtype0*
_output_shapes

: 

й
0classification_layers/dense0/dense/kernel/AssignAssign)classification_layers/dense0/dense/kernelFclassification_layers/dense0/dense/kernel/Initializer/truncated_normal*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
validate_shape(*
use_locking(
╠
.classification_layers/dense0/dense/kernel/readIdentity)classification_layers/dense0/dense/kernel*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 

┬
9classification_layers/dense0/dense/bias/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
_output_shapes
:
*
dtype0
¤
'classification_layers/dense0/dense/bias
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
dtype0*
	container 
д
.classification_layers/dense0/dense/bias/AssignAssign'classification_layers/dense0/dense/bias9classification_layers/dense0/dense/bias/Initializer/zeros*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
┬
,classification_layers/dense0/dense/bias/readIdentity'classification_layers/dense0/dense/bias*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0
╠
)classification_layers/dense0/dense/MatMulMatMulFlatten/Reshape.classification_layers/dense0/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         
*
transpose_a( 
О
*classification_layers/dense0/dense/BiasAddBiasAdd)classification_layers/dense0/dense/MatMul,classification_layers/dense0/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         

Є
!classification_layers/dense0/ReluRelu*classification_layers/dense0/dense/BiasAdd*
T0*'
_output_shapes
:         

І
*classification_layers/dense0/dropout/ShapeShape!classification_layers/dense0/Relu*
out_type0*
_output_shapes
:*
T0
|
7classification_layers/dense0/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
|
7classification_layers/dense0/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
о
Aclassification_layers/dense0/dropout/random_uniform/RandomUniformRandomUniform*classification_layers/dense0/dropout/Shape*'
_output_shapes
:         
*
seed2 *
T0*

seed *
dtype0
Л
7classification_layers/dense0/dropout/random_uniform/subSub7classification_layers/dense0/dropout/random_uniform/max7classification_layers/dense0/dropout/random_uniform/min*
T0*
_output_shapes
: 
В
7classification_layers/dense0/dropout/random_uniform/mulMulAclassification_layers/dense0/dropout/random_uniform/RandomUniform7classification_layers/dense0/dropout/random_uniform/sub*'
_output_shapes
:         
*
T0
я
3classification_layers/dense0/dropout/random_uniformAdd7classification_layers/dense0/dropout/random_uniform/mul7classification_layers/dense0/dropout/random_uniform/min*'
_output_shapes
:         
*
T0
ф
(classification_layers/dense0/dropout/addAdd!classification_layers/Placeholder3classification_layers/dense0/dropout/random_uniform*
T0*
_output_shapes
:
ђ
*classification_layers/dense0/dropout/FloorFloor(classification_layers/dense0/dropout/add*
T0*
_output_shapes
:
ю
(classification_layers/dense0/dropout/divRealDiv!classification_layers/dense0/Relu!classification_layers/Placeholder*
_output_shapes
:*
T0
и
(classification_layers/dense0/dropout/mulMul(classification_layers/dense0/dropout/div*classification_layers/dense0/dropout/Floor*
T0*'
_output_shapes
:         

с
Pclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shapeConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB"
      *
dtype0*
_output_shapes
:
о
Oclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/meanConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
п
Qclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddevConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
л
Zclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalPclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *
dtype0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

№
Nclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulMulZclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalQclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddev*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

П
Jclassification_layers/dense_last/dense/kernel/Initializer/truncated_normalAddNclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulOclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mean*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

с
-classification_layers/dense_last/dense/kernel
VariableV2*
shared_name *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
	container *
shape
:
*
dtype0*
_output_shapes

:

═
4classification_layers/dense_last/dense/kernel/AssignAssign-classification_layers/dense_last/dense/kernelJclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
п
2classification_layers/dense_last/dense/kernel/readIdentity-classification_layers/dense_last/dense/kernel*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

╩
=classification_layers/dense_last/dense/bias/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
О
+classification_layers/dense_last/dense/bias
VariableV2*
	container *
dtype0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
shared_name 
Х
2classification_layers/dense_last/dense/bias/AssignAssign+classification_layers/dense_last/dense/bias=classification_layers/dense_last/dense/bias/Initializer/zeros*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╬
0classification_layers/dense_last/dense/bias/readIdentity+classification_layers/dense_last/dense/bias*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:
ь
-classification_layers/dense_last/dense/MatMulMatMul(classification_layers/dense0/dropout/mul2classification_layers/dense_last/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
с
.classification_layers/dense_last/dense/BiasAddBiasAdd-classification_layers/dense_last/dense/MatMul0classification_layers/dense_last/dense/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
і
classification_layers/SoftmaxSoftmax.classification_layers/dense_last/dense/BiasAdd*
T0*'
_output_shapes
:         
n
)Evaluation_layers/clip_by_value/Minimum/yConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
«
'Evaluation_layers/clip_by_value/MinimumMinimumclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*'
_output_shapes
:         *
T0
f
!Evaluation_layers/clip_by_value/yConst*
valueB
 * Т█.*
dtype0*
_output_shapes
: 
е
Evaluation_layers/clip_by_valueMaximum'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:         
o
Evaluation_layers/LogLogEvaluation_layers/clip_by_value*
T0*'
_output_shapes
:         
y
Evaluation_layers/mulMulTarget/PlaceholderEvaluation_layers/Log*
T0*'
_output_shapes
:         
q
'Evaluation_layers/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Д
Evaluation_layers/SumSumEvaluation_layers/mul'Evaluation_layers/Sum/reduction_indices*#
_output_shapes
:         *
T0*
	keep_dims( *

Tidx0
a
Evaluation_layers/NegNegEvaluation_layers/Sum*
T0*#
_output_shapes
:         
a
Evaluation_layers/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ї
Evaluation_layers/MeanMeanEvaluation_layers/NegEvaluation_layers/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
d
"Evaluation_layers/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
Ъ
Evaluation_layers/ArgMaxArgMaxclassification_layers/Softmax"Evaluation_layers/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:         
f
$Evaluation_layers/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
ў
Evaluation_layers/ArgMax_1ArgMaxTarget/Placeholder$Evaluation_layers/ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:         
ё
Evaluation_layers/EqualEqualEvaluation_layers/ArgMaxEvaluation_layers/ArgMax_1*#
_output_shapes
:         *
T0	
|
Evaluation_layers/accracy/CastCastEvaluation_layers/Equal*

SrcT0
*#
_output_shapes
:         *

DstT0
i
Evaluation_layers/accracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Evaluation_layers/accracy/MeanMeanEvaluation_layers/accracy/CastEvaluation_layers/accracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
z
Evaluation_layers/accuracy/tagsConst*+
value"B  BEvaluation_layers/accuracy*
_output_shapes
: *
dtype0
Ї
Evaluation_layers/accuracyScalarSummaryEvaluation_layers/accuracy/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
r
Evaluation_layers/loss/tagsConst*'
valueB BEvaluation_layers/loss*
dtype0*
_output_shapes
: 
}
Evaluation_layers/lossScalarSummaryEvaluation_layers/loss/tagsEvaluation_layers/Mean*
_output_shapes
: *
T0
~
!Evaluation_layers/accuracy_1/tagsConst*-
value$B" BEvaluation_layers/accuracy_1*
_output_shapes
: *
dtype0
Љ
Evaluation_layers/accuracy_1ScalarSummary!Evaluation_layers/accuracy_1/tagsEvaluation_layers/accracy/Mean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
}
3gradients/Evaluation_layers/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
░
-gradients/Evaluation_layers/Mean_grad/ReshapeReshapegradients/Fill3gradients/Evaluation_layers/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
ђ
+gradients/Evaluation_layers/Mean_grad/ShapeShapeEvaluation_layers/Neg*
T0*
out_type0*
_output_shapes
:
╬
*gradients/Evaluation_layers/Mean_grad/TileTile-gradients/Evaluation_layers/Mean_grad/Reshape+gradients/Evaluation_layers/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
ѓ
-gradients/Evaluation_layers/Mean_grad/Shape_1ShapeEvaluation_layers/Neg*
out_type0*
_output_shapes
:*
T0
p
-gradients/Evaluation_layers/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
u
+gradients/Evaluation_layers/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
╠
*gradients/Evaluation_layers/Mean_grad/ProdProd-gradients/Evaluation_layers/Mean_grad/Shape_1+gradients/Evaluation_layers/Mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
w
-gradients/Evaluation_layers/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
л
,gradients/Evaluation_layers/Mean_grad/Prod_1Prod-gradients/Evaluation_layers/Mean_grad/Shape_2-gradients/Evaluation_layers/Mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
q
/gradients/Evaluation_layers/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
И
-gradients/Evaluation_layers/Mean_grad/MaximumMaximum,gradients/Evaluation_layers/Mean_grad/Prod_1/gradients/Evaluation_layers/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Х
.gradients/Evaluation_layers/Mean_grad/floordivFloorDiv*gradients/Evaluation_layers/Mean_grad/Prod-gradients/Evaluation_layers/Mean_grad/Maximum*
_output_shapes
: *
T0
њ
*gradients/Evaluation_layers/Mean_grad/CastCast.gradients/Evaluation_layers/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Й
-gradients/Evaluation_layers/Mean_grad/truedivRealDiv*gradients/Evaluation_layers/Mean_grad/Tile*gradients/Evaluation_layers/Mean_grad/Cast*
T0*#
_output_shapes
:         
ї
(gradients/Evaluation_layers/Neg_grad/NegNeg-gradients/Evaluation_layers/Mean_grad/truediv*#
_output_shapes
:         *
T0

*gradients/Evaluation_layers/Sum_grad/ShapeShapeEvaluation_layers/mul*
T0*
out_type0*
_output_shapes
:
k
)gradients/Evaluation_layers/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
е
(gradients/Evaluation_layers/Sum_grad/addAdd'Evaluation_layers/Sum/reduction_indices)gradients/Evaluation_layers/Sum_grad/Size*
T0*
_output_shapes
:
«
(gradients/Evaluation_layers/Sum_grad/modFloorMod(gradients/Evaluation_layers/Sum_grad/add)gradients/Evaluation_layers/Sum_grad/Size*
T0*
_output_shapes
:
v
,gradients/Evaluation_layers/Sum_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
r
0gradients/Evaluation_layers/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
r
0gradients/Evaluation_layers/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ж
*gradients/Evaluation_layers/Sum_grad/rangeRange0gradients/Evaluation_layers/Sum_grad/range/start)gradients/Evaluation_layers/Sum_grad/Size0gradients/Evaluation_layers/Sum_grad/range/delta*
_output_shapes
:*

Tidx0
q
/gradients/Evaluation_layers/Sum_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
х
)gradients/Evaluation_layers/Sum_grad/FillFill,gradients/Evaluation_layers/Sum_grad/Shape_1/gradients/Evaluation_layers/Sum_grad/Fill/value*
_output_shapes
:*
T0
Д
2gradients/Evaluation_layers/Sum_grad/DynamicStitchDynamicStitch*gradients/Evaluation_layers/Sum_grad/range(gradients/Evaluation_layers/Sum_grad/mod*gradients/Evaluation_layers/Sum_grad/Shape)gradients/Evaluation_layers/Sum_grad/Fill*
T0*
N*#
_output_shapes
:         
p
.gradients/Evaluation_layers/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
╔
,gradients/Evaluation_layers/Sum_grad/MaximumMaximum2gradients/Evaluation_layers/Sum_grad/DynamicStitch.gradients/Evaluation_layers/Sum_grad/Maximum/y*
T0*#
_output_shapes
:         
И
-gradients/Evaluation_layers/Sum_grad/floordivFloorDiv*gradients/Evaluation_layers/Sum_grad/Shape,gradients/Evaluation_layers/Sum_grad/Maximum*
T0*
_output_shapes
:
к
,gradients/Evaluation_layers/Sum_grad/ReshapeReshape(gradients/Evaluation_layers/Neg_grad/Neg2gradients/Evaluation_layers/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
м
)gradients/Evaluation_layers/Sum_grad/TileTile,gradients/Evaluation_layers/Sum_grad/Reshape-gradients/Evaluation_layers/Sum_grad/floordiv*'
_output_shapes
:         *
T0*

Tmultiples0
|
*gradients/Evaluation_layers/mul_grad/ShapeShapeTarget/Placeholder*
T0*
out_type0*
_output_shapes
:
Ђ
,gradients/Evaluation_layers/mul_grad/Shape_1ShapeEvaluation_layers/Log*
T0*
out_type0*
_output_shapes
:
Ж
:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Evaluation_layers/mul_grad/Shape,gradients/Evaluation_layers/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Б
(gradients/Evaluation_layers/mul_grad/mulMul)gradients/Evaluation_layers/Sum_grad/TileEvaluation_layers/Log*'
_output_shapes
:         *
T0
Н
(gradients/Evaluation_layers/mul_grad/SumSum(gradients/Evaluation_layers/mul_grad/mul:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
═
,gradients/Evaluation_layers/mul_grad/ReshapeReshape(gradients/Evaluation_layers/mul_grad/Sum*gradients/Evaluation_layers/mul_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
б
*gradients/Evaluation_layers/mul_grad/mul_1MulTarget/Placeholder)gradients/Evaluation_layers/Sum_grad/Tile*'
_output_shapes
:         *
T0
█
*gradients/Evaluation_layers/mul_grad/Sum_1Sum*gradients/Evaluation_layers/mul_grad/mul_1<gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
М
.gradients/Evaluation_layers/mul_grad/Reshape_1Reshape*gradients/Evaluation_layers/mul_grad/Sum_1,gradients/Evaluation_layers/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ю
5gradients/Evaluation_layers/mul_grad/tuple/group_depsNoOp-^gradients/Evaluation_layers/mul_grad/Reshape/^gradients/Evaluation_layers/mul_grad/Reshape_1
б
=gradients/Evaluation_layers/mul_grad/tuple/control_dependencyIdentity,gradients/Evaluation_layers/mul_grad/Reshape6^gradients/Evaluation_layers/mul_grad/tuple/group_deps*?
_class5
31loc:@gradients/Evaluation_layers/mul_grad/Reshape*'
_output_shapes
:         *
T0
е
?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1Identity.gradients/Evaluation_layers/mul_grad/Reshape_16^gradients/Evaluation_layers/mul_grad/tuple/group_deps*A
_class7
53loc:@gradients/Evaluation_layers/mul_grad/Reshape_1*'
_output_shapes
:         *
T0
м
/gradients/Evaluation_layers/Log_grad/Reciprocal
ReciprocalEvaluation_layers/clip_by_value@^gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
М
(gradients/Evaluation_layers/Log_grad/mulMul?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1/gradients/Evaluation_layers/Log_grad/Reciprocal*
T0*'
_output_shapes
:         
Џ
4gradients/Evaluation_layers/clip_by_value_grad/ShapeShape'Evaluation_layers/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
y
6gradients/Evaluation_layers/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ъ
6gradients/Evaluation_layers/clip_by_value_grad/Shape_2Shape(gradients/Evaluation_layers/Log_grad/mul*
out_type0*
_output_shapes
:*
T0

:gradients/Evaluation_layers/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Р
4gradients/Evaluation_layers/clip_by_value_grad/zerosFill6gradients/Evaluation_layers/clip_by_value_grad/Shape_2:gradients/Evaluation_layers/clip_by_value_grad/zeros/Const*
T0*'
_output_shapes
:         
╔
;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqualGreaterEqual'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:         
ѕ
Dgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/Evaluation_layers/clip_by_value_grad/Shape6gradients/Evaluation_layers/clip_by_value_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ј
5gradients/Evaluation_layers/clip_by_value_grad/SelectSelect;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:         
Г
9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot
LogicalNot;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual*'
_output_shapes
:         
ј
7gradients/Evaluation_layers/clip_by_value_grad/Select_1Select9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*'
_output_shapes
:         *
T0
Ш
2gradients/Evaluation_layers/clip_by_value_grad/SumSum5gradients/Evaluation_layers/clip_by_value_grad/SelectDgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
в
6gradients/Evaluation_layers/clip_by_value_grad/ReshapeReshape2gradients/Evaluation_layers/clip_by_value_grad/Sum4gradients/Evaluation_layers/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ч
4gradients/Evaluation_layers/clip_by_value_grad/Sum_1Sum7gradients/Evaluation_layers/clip_by_value_grad/Select_1Fgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Я
8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1Reshape4gradients/Evaluation_layers/clip_by_value_grad/Sum_16gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
╗
?gradients/Evaluation_layers/clip_by_value_grad/tuple/group_depsNoOp7^gradients/Evaluation_layers/clip_by_value_grad/Reshape9^gradients/Evaluation_layers/clip_by_value_grad/Reshape_1
╩
Ggradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependencyIdentity6gradients/Evaluation_layers/clip_by_value_grad/Reshape@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape*'
_output_shapes
:         
┐
Igradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency_1Identity8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape_1*
_output_shapes
: 
Ў
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/ShapeShapeclassification_layers/Softmax*
T0*
out_type0*
_output_shapes
:
Ђ
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
┼
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2ShapeGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Є
Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Щ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zerosFill>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:         *
T0
╔
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual	LessEqualclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*'
_output_shapes
:         *
T0
а
Lgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┬
=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectSelect@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqualGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:         *
T0
║
Agradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNot
LogicalNot@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual*'
_output_shapes
:         
┼
?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1SelectAgradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNotGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:         *
T0
ј
:gradients/Evaluation_layers/clip_by_value/Minimum_grad/SumSum=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectLgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ѓ
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeReshape:gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ћ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1Sum?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1Ngradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1Reshape<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
М
Ggradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_depsNoOp?^gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeA^gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1
Ж
Ogradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeH^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:         *
T0
▀
Qgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1H^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
┘
0gradients/classification_layers/Softmax_grad/mulMulOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyclassification_layers/Softmax*
T0*'
_output_shapes
:         
ї
Bgradients/classification_layers/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Э
0gradients/classification_layers/Softmax_grad/SumSum0gradients/classification_layers/Softmax_grad/mulBgradients/classification_layers/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
І
:gradients/classification_layers/Softmax_grad/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ь
4gradients/classification_layers/Softmax_grad/ReshapeReshape0gradients/classification_layers/Softmax_grad/Sum:gradients/classification_layers/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
­
0gradients/classification_layers/Softmax_grad/subSubOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency4gradients/classification_layers/Softmax_grad/Reshape*
T0*'
_output_shapes
:         
╝
2gradients/classification_layers/Softmax_grad/mul_1Mul0gradients/classification_layers/Softmax_grad/subclassification_layers/Softmax*
T0*'
_output_shapes
:         
╚
Igradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/classification_layers/Softmax_grad/mul_1*
_output_shapes
:*
T0*
data_formatNHWC
О
Ngradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/classification_layers/Softmax_grad/mul_1J^gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad
Я
Vgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/classification_layers/Softmax_grad/mul_1O^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/classification_layers/Softmax_grad/mul_1*'
_output_shapes
:         
Ѓ
Xgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradO^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
▒
Cgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulMatMulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency2classification_layers/dense_last/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:         
*
transpose_a( 
а
Egradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1MatMul(classification_layers/dense0/dropout/mulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
с
Mgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_depsNoOpD^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMulF^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1
ђ
Ugradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependencyIdentityCgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulN^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul*'
_output_shapes
:         
*
T0
§
Wgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1N^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1*
_output_shapes

:

«
=gradients/classification_layers/dense0/dropout/mul_grad/ShapeShape(classification_layers/dense0/dropout/div*
T0*
out_type0*#
_output_shapes
:         
▓
?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1Shape*classification_layers/dense0/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
Б
Mgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/mul_grad/Shape?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
У
;gradients/classification_layers/dense0/dropout/mul_grad/mulMulUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*classification_layers/dense0/dropout/Floor*
T0*
_output_shapes
:
ј
;gradients/classification_layers/dense0/dropout/mul_grad/SumSum;gradients/classification_layers/dense0/dropout/mul_grad/mulMgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
э
?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/mul_grad/Sum=gradients/classification_layers/dense0/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
У
=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Mul(classification_layers/dense0/dropout/divUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
ћ
=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1Sum=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Ogradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
§
Agradients/classification_layers/dense0/dropout/mul_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
о
Hgradients/classification_layers/dense0/dropout/mul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/mul_grad/ReshapeB^gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1
▀
Pgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeI^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape*
_output_shapes
:
т
Rgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/mul_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
ъ
=gradients/classification_layers/dense0/dropout/div_grad/ShapeShape!classification_layers/dense0/Relu*
out_type0*
_output_shapes
:*
T0
Е
?gradients/classification_layers/dense0/dropout/div_grad/Shape_1Shape!classification_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
Б
Mgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/div_grad/Shape?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
?gradients/classification_layers/dense0/dropout/div_grad/RealDivRealDivPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency!classification_layers/Placeholder*
_output_shapes
:*
T0
њ
;gradients/classification_layers/dense0/dropout/div_grad/SumSum?gradients/classification_layers/dense0/dropout/div_grad/RealDivMgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
є
?gradients/classification_layers/dense0/dropout/div_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/div_grad/Sum=gradients/classification_layers/dense0/dropout/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

Ќ
;gradients/classification_layers/dense0/dropout/div_grad/NegNeg!classification_layers/dense0/Relu*'
_output_shapes
:         
*
T0
¤
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_1RealDiv;gradients/classification_layers/dense0/dropout/div_grad/Neg!classification_layers/Placeholder*
_output_shapes
:*
T0
Н
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_2RealDivAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_1!classification_layers/Placeholder*
T0*
_output_shapes
:
Щ
;gradients/classification_layers/dense0/dropout/div_grad/mulMulPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
њ
=gradients/classification_layers/dense0/dropout/div_grad/Sum_1Sum;gradients/classification_layers/dense0/dropout/div_grad/mulOgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
§
Agradients/classification_layers/dense0/dropout/div_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/div_grad/Sum_1?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
о
Hgradients/classification_layers/dense0/dropout/div_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/div_grad/ReshapeB^gradients/classification_layers/dense0/dropout/div_grad/Reshape_1
Ь
Pgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/div_grad/ReshapeI^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape*'
_output_shapes
:         
*
T0
т
Rgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/div_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape_1*
_output_shapes
:
В
9gradients/classification_layers/dense0/Relu_grad/ReluGradReluGradPgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency!classification_layers/dense0/Relu*
T0*'
_output_shapes
:         

╦
Egradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/classification_layers/dense0/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

о
Jgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_depsNoOp:^gradients/classification_layers/dense0/Relu_grad/ReluGradF^gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad
Т
Rgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/classification_layers/dense0/Relu_grad/ReluGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/classification_layers/dense0/Relu_grad/ReluGrad*'
_output_shapes
:         
*
T0
з
Tgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Ц
?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulMatMulRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency.classification_layers/dense0/dense/kernel/read*
transpose_b(*'
_output_shapes
:          *
transpose_a( *
T0
 
Agradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1MatMulFlatten/ReshapeRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

: 
*
transpose_a(
О
Igradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dense/MatMul_grad/MatMulB^gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1
­
Qgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulJ^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul*'
_output_shapes
:          *
T0
ь
Sgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1J^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1*
_output_shapes

: 
*
T0
Ј
$gradients/Flatten/Reshape_grad/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/Squeeze*
T0*
out_type0*
_output_shapes
:
Ь
&gradients/Flatten/Reshape_grad/ReshapeReshapeQgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency$gradients/Flatten/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ф
@gradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/MaxPool*
T0*
out_type0*
_output_shapes
:
 
Bgradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/ReshapeReshape&gradients/Flatten/Reshape_grad/Reshape@gradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
Ѓ
Fgradients/conv1D_layers/conv1d9/max_pooling1d/MaxPool_grad/MaxPoolGradMaxPoolGrad.conv1D_layers/conv1d9/max_pooling1d/ExpandDims+conv1D_layers/conv1d9/max_pooling1d/MaxPoolBgradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:         
ц
Cgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d8/dropout/mul*
out_type0*
_output_shapes
:*
T0
А
Egradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/ReshapeReshapeFgradients/conv1D_layers/conv1d9/max_pooling1d/MaxPool_grad/MaxPoolGradCgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
а
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d8/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d8/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╩
4gradients/conv1D_layers/conv1d8/dropout/mul_grad/mulMulEgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Reshape#conv1D_layers/conv1d8/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d8/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d8/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d8/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
╩
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d8/dropout/divEgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d8/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d8/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1*
_output_shapes
:
Ќ
6gradients/conv1D_layers/conv1d8/dropout/div_grad/ShapeShape!conv1D_layers/conv1d8/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d8/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d8/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d8/dropout/div_grad/NegNeg!conv1D_layers/conv1d8/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d8/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d8/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d8/dropout/div_grad/mulHgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d8/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d8/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d8/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4                                    *
data_formatNHWC*
strides

Ъ
Fgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d8/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d7/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d7/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d7/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d7/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d7/dropout/Floor*
T0*
_output_shapes
:
щ
4gradients/conv1D_layers/conv1d7/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d7/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d7/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¤
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d7/dropout/divJgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d7/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d7/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d7/dropout/div_grad/ShapeShape!conv1D_layers/conv1d7/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d7/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d7/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d7/dropout/div_grad/NegNeg!conv1D_layers/conv1d7/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d7/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d7/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d7/dropout/div_grad/mulHgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d7/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape*+
_output_shapes
:         *
T0
╔
Kgradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d7/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d7/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Reshape*J
_output_shapes8
6:4                                    *
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
Ъ
Fgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d7/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d6/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/Shape*
Tshape0*"
_output_shapes
:*
T0
а
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d6/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d6/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d6/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d6/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d6/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d6/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d6/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¤
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d6/dropout/divJgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d6/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d6/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d6/dropout/div_grad/ShapeShape!conv1D_layers/conv1d6/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d6/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d6/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d6/dropout/div_grad/NegNeg!conv1D_layers/conv1d6/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d6/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d6/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d6/dropout/div_grad/mulHgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d6/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape*+
_output_shapes
:         *
T0
╔
Kgradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d6/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d6/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Reshape*J
_output_shapes8
6:4                                    *
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
Ъ
Fgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d6/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d5/dropout/mul*
out_type0*
_output_shapes
:*
T0
╝
Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/Shape*
Tshape0*"
_output_shapes
:*
T0
а
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d5/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d5/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d5/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d5/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d5/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d5/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d5/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d5/dropout/divJgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d5/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d5/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d5/dropout/div_grad/ShapeShape!conv1D_layers/conv1d5/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d5/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d5/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d5/dropout/div_grad/NegNeg!conv1D_layers/conv1d5/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d5/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d5/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d5/dropout/div_grad/mulHgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d5/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape*+
_output_shapes
:         *
T0
╔
Kgradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d5/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d5/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Reshape*J
_output_shapes8
6:4                                    *
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
Ъ
Fgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d5/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d4/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d4/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d4/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d4/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d4/dropout/Floor*
T0*
_output_shapes
:
щ
4gradients/conv1D_layers/conv1d4/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d4/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d4/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¤
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d4/dropout/divJgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d4/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d4/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d4/dropout/div_grad/ShapeShape!conv1D_layers/conv1d4/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d4/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d4/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d4/dropout/div_grad/NegNeg!conv1D_layers/conv1d4/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d4/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d4/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d4/dropout/div_grad/mulHgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d4/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d4/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d4/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4                                    *
data_formatNHWC*
strides

Ъ
Fgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d4/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d3/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d3/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d3/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d3/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d3/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d3/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d3/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d3/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d3/dropout/divJgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d3/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d3/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d3/dropout/div_grad/ShapeShape!conv1D_layers/conv1d3/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d3/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ш
8gradients/conv1D_layers/conv1d3/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d3/dropout/div_grad/NegNeg!conv1D_layers/conv1d3/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d3/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d3/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d3/dropout/div_grad/mulHgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d3/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d3/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d3/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d3/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:*
data_formatNHWC*
strides

ѓ
Ogradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d2/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d2/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d2/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d2/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d2/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d2/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d2/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d2/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d2/dropout/divJgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d2/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d2/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1*
_output_shapes
:
Ќ
6gradients/conv1D_layers/conv1d2/dropout/div_grad/ShapeShape!conv1D_layers/conv1d2/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d2/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d2/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d2/dropout/div_grad/NegNeg!conv1D_layers/conv1d2/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d2/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d2/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d2/dropout/div_grad/mulHgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d2/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape*+
_output_shapes
:         *
T0
╔
Kgradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d2/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d2/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d2/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d1/dropout/mul*
out_type0*
_output_shapes
:*
T0
╝
Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d1/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d1/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d1/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d1/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d1/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d1/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d1/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¤
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d1/dropout/divJgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d1/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d1/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d1/dropout/div_grad/ShapeShape!conv1D_layers/conv1d1/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d1/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d1/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d1/dropout/div_grad/NegNeg!conv1D_layers/conv1d1/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d1/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d1/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d1/dropout/div_grad/mulHgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d1/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d1/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d1/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d1/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:          
Ц
Ygradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Ъ
Jgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
ќ
beta1_power/initial_valueConst*
valueB
 *fff?*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
_output_shapes
: 
Д
beta1_power
VariableV2*
shape: *
_output_shapes
: *
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
	container 
к
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
ѓ
beta1_power/readIdentitybeta1_power*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
ќ
beta2_power/initial_valueConst*
valueB
 *wЙ?*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
_output_shapes
: 
Д
beta2_power
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
shape: *
shared_name 
к
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
validate_shape(*
_output_shapes
: 
ѓ
beta2_power/readIdentitybeta2_power*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
¤
:conv1D_layers/conv1d1/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d1/conv1d/kernel/Adam
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
dtype0*
	container 
Г
/conv1D_layers/conv1d1/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d1/conv1d/kernel/Adam:conv1D_layers/conv1d1/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d1/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d1/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d1/conv1d/kernel/Adam_1
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
│
1conv1D_layers/conv1d1/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d1/conv1d/kernel/Adam_1<conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d1/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d1/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d1/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d1/conv1d/bias/Adam
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
-conv1D_layers/conv1d1/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d1/conv1d/bias/Adam8conv1D_layers/conv1d1/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d1/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d1/conv1d/bias/Adam*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:
й
:conv1D_layers/conv1d1/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d1/conv1d/bias/Adam_1
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Б
/conv1D_layers/conv1d1/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d1/conv1d/bias/Adam_1:conv1D_layers/conv1d1/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d1/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d1/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0
¤
:conv1D_layers/conv1d2/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d2/conv1d/kernel/Adam
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Г
/conv1D_layers/conv1d2/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d2/conv1d/kernel/Adam:conv1D_layers/conv1d2/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d2/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d2/conv1d/kernel/Adam*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0
Л
<conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d2/conv1d/kernel/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
│
1conv1D_layers/conv1d2/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d2/conv1d/kernel/Adam_1<conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d2/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d2/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d2/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d2/conv1d/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
dtype0*
	container 
Ю
-conv1D_layers/conv1d2/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d2/conv1d/bias/Adam8conv1D_layers/conv1d2/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d2/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d2/conv1d/bias/Adam*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:
й
:conv1D_layers/conv1d2/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d2/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d2/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d2/conv1d/bias/Adam_1:conv1D_layers/conv1d2/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d2/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d2/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d3/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d3/conv1d/kernel/Adam
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Г
/conv1D_layers/conv1d3/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d3/conv1d/kernel/Adam:conv1D_layers/conv1d3/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d3/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d3/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d3/conv1d/kernel/Adam_1
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
dtype0*
	container 
│
1conv1D_layers/conv1d3/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d3/conv1d/kernel/Adam_1<conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d3/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d3/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d3/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d3/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d3/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d3/conv1d/bias/Adam8conv1D_layers/conv1d3/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d3/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d3/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d3/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d3/conv1d/bias/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
dtype0*
	container 
Б
/conv1D_layers/conv1d3/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d3/conv1d/bias/Adam_1:conv1D_layers/conv1d3/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d3/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d3/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d4/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d4/conv1d/kernel/Adam
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Г
/conv1D_layers/conv1d4/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d4/conv1d/kernel/Adam:conv1D_layers/conv1d4/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d4/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d4/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d4/conv1d/kernel/Adam_1
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
dtype0*
	container 
│
1conv1D_layers/conv1d4/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d4/conv1d/kernel/Adam_1<conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d4/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d4/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d4/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d4/conv1d/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
dtype0*
	container 
Ю
-conv1D_layers/conv1d4/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d4/conv1d/bias/Adam8conv1D_layers/conv1d4/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d4/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d4/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d4/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d4/conv1d/bias/Adam_1
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Б
/conv1D_layers/conv1d4/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d4/conv1d/bias/Adam_1:conv1D_layers/conv1d4/conv1d/bias/Adam_1/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
-conv1D_layers/conv1d4/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d4/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d5/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d5/conv1d/kernel/Adam
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Г
/conv1D_layers/conv1d5/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d5/conv1d/kernel/Adam:conv1D_layers/conv1d5/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d5/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d5/conv1d/kernel/Adam*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
Л
<conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d5/conv1d/kernel/Adam_1
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
dtype0*
	container 
│
1conv1D_layers/conv1d5/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d5/conv1d/kernel/Adam_1<conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d5/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d5/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d5/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d5/conv1d/bias/Adam
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Ю
-conv1D_layers/conv1d5/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d5/conv1d/bias/Adam8conv1D_layers/conv1d5/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d5/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d5/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d5/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d5/conv1d/bias/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
dtype0*
	container 
Б
/conv1D_layers/conv1d5/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d5/conv1d/bias/Adam_1:conv1D_layers/conv1d5/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d5/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d5/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0
¤
:conv1D_layers/conv1d6/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d6/conv1d/kernel/Adam
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Г
/conv1D_layers/conv1d6/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d6/conv1d/kernel/Adam:conv1D_layers/conv1d6/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d6/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d6/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d6/conv1d/kernel/Adam_1
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
│
1conv1D_layers/conv1d6/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d6/conv1d/kernel/Adam_1<conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d6/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d6/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d6/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d6/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d6/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d6/conv1d/bias/Adam8conv1D_layers/conv1d6/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d6/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d6/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d6/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d6/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d6/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d6/conv1d/bias/Adam_1:conv1D_layers/conv1d6/conv1d/bias/Adam_1/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
-conv1D_layers/conv1d6/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d6/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d7/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d7/conv1d/kernel/Adam
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Г
/conv1D_layers/conv1d7/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d7/conv1d/kernel/Adam:conv1D_layers/conv1d7/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d7/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d7/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d7/conv1d/kernel/Adam_1
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
│
1conv1D_layers/conv1d7/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d7/conv1d/kernel/Adam_1<conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d7/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d7/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d7/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d7/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d7/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d7/conv1d/bias/Adam8conv1D_layers/conv1d7/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d7/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d7/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d7/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d7/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d7/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d7/conv1d/bias/Adam_1:conv1D_layers/conv1d7/conv1d/bias/Adam_1/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
-conv1D_layers/conv1d7/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d7/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d8/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d8/conv1d/kernel/Adam
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Г
/conv1D_layers/conv1d8/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d8/conv1d/kernel/Adam:conv1D_layers/conv1d8/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d8/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d8/conv1d/kernel/Adam*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0
Л
<conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d8/conv1d/kernel/Adam_1
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
│
1conv1D_layers/conv1d8/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d8/conv1d/kernel/Adam_1<conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d8/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d8/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d8/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d8/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d8/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d8/conv1d/bias/Adam8conv1D_layers/conv1d8/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d8/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d8/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d8/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d8/conv1d/bias/Adam_1
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Б
/conv1D_layers/conv1d8/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d8/conv1d/bias/Adam_1:conv1D_layers/conv1d8/conv1d/bias/Adam_1/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
-conv1D_layers/conv1d8/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d8/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0
М
@classification_layers/dense0/dense/kernel/Adam/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB 
*    *
_output_shapes

: 
*
dtype0
Я
.classification_layers/dense0/dense/kernel/Adam
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
shape
: 
*
shared_name 
┴
5classification_layers/dense0/dense/kernel/Adam/AssignAssign.classification_layers/dense0/dense/kernel/Adam@classification_layers/dense0/dense/kernel/Adam/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
validate_shape(*
use_locking(
о
3classification_layers/dense0/dense/kernel/Adam/readIdentity.classification_layers/dense0/dense/kernel/Adam*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
Н
Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB 
*    *
_output_shapes

: 
*
dtype0
Р
0classification_layers/dense0/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
shape
: 
*
shared_name 
К
7classification_layers/dense0/dense/kernel/Adam_1/AssignAssign0classification_layers/dense0/dense/kernel/Adam_1Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
validate_shape(*
use_locking(
┌
5classification_layers/dense0/dense/kernel/Adam_1/readIdentity0classification_layers/dense0/dense/kernel/Adam_1*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 

К
>classification_layers/dense0/dense/bias/Adam/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
_output_shapes
:
*
dtype0
н
,classification_layers/dense0/dense/bias/Adam
VariableV2*
	container *
dtype0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
shape:
*
shared_name 
х
3classification_layers/dense0/dense/bias/Adam/AssignAssign,classification_layers/dense0/dense/bias/Adam>classification_layers/dense0/dense/bias/Adam/Initializer/zeros*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
╠
1classification_layers/dense0/dense/bias/Adam/readIdentity,classification_layers/dense0/dense/bias/Adam*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0
╔
@classification_layers/dense0/dense/bias/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
dtype0*
_output_shapes
:

о
.classification_layers/dense0/dense/bias/Adam_1
VariableV2*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
╗
5classification_layers/dense0/dense/bias/Adam_1/AssignAssign.classification_layers/dense0/dense/bias/Adam_1@classification_layers/dense0/dense/bias/Adam_1/Initializer/zeros*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
л
3classification_layers/dense0/dense/bias/Adam_1/readIdentity.classification_layers/dense0/dense/bias/Adam_1*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:

█
Dclassification_layers/dense_last/dense/kernel/Adam/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

У
2classification_layers/dense_last/dense/kernel/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
shared_name *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
dtype0*
	container 
Л
9classification_layers/dense_last/dense/kernel/Adam/AssignAssign2classification_layers/dense_last/dense/kernel/AdamDclassification_layers/dense_last/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
validate_shape(*
_output_shapes

:

Р
7classification_layers/dense_last/dense/kernel/Adam/readIdentity2classification_layers/dense_last/dense/kernel/Adam*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
П
Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

Ж
4classification_layers/dense_last/dense/kernel/Adam_1
VariableV2*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
shape
:
*
dtype0*
shared_name *
	container 
О
;classification_layers/dense_last/dense/kernel/Adam_1/AssignAssign4classification_layers/dense_last/dense/kernel/Adam_1Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
validate_shape(*
_output_shapes

:

Т
9classification_layers/dense_last/dense/kernel/Adam_1/readIdentity4classification_layers/dense_last/dense/kernel/Adam_1*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

¤
Bclassification_layers/dense_last/dense/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
▄
0classification_layers/dense_last/dense/bias/Adam
VariableV2*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
┼
7classification_layers/dense_last/dense/bias/Adam/AssignAssign0classification_layers/dense_last/dense/bias/AdamBclassification_layers/dense_last/dense/bias/Adam/Initializer/zeros*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
п
5classification_layers/dense_last/dense/bias/Adam/readIdentity0classification_layers/dense_last/dense/bias/Adam*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0
Л
Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
я
2classification_layers/dense_last/dense/bias/Adam_1
VariableV2*
	container *
dtype0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
shared_name 
╦
9classification_layers/dense_last/dense/bias/Adam_1/AssignAssign2classification_layers/dense_last/dense/bias/Adam_1Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zeros*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
▄
7classification_layers/dense_last/dense/bias/Adam_1/readIdentity2classification_layers/dense_last/dense/bias/Adam_1*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *oЃ:*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wЙ?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
щ
9Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d1/conv1d/kernel(conv1D_layers/conv1d1/conv1d/kernel/Adam*conv1D_layers/conv1d1/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d1/conv1d/bias&conv1D_layers/conv1d1/conv1d/bias/Adam(conv1D_layers/conv1d1/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
щ
9Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d2/conv1d/kernel(conv1D_layers/conv1d2/conv1d/kernel/Adam*conv1D_layers/conv1d2/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d2/conv1d/bias&conv1D_layers/conv1d2/conv1d/bias/Adam(conv1D_layers/conv1d2/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d3/conv1d/kernel(conv1D_layers/conv1d3/conv1d/kernel/Adam*conv1D_layers/conv1d3/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d3/conv1d/bias&conv1D_layers/conv1d3/conv1d/bias/Adam(conv1D_layers/conv1d3/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d4/conv1d/kernel(conv1D_layers/conv1d4/conv1d/kernel/Adam*conv1D_layers/conv1d4/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d4/conv1d/bias&conv1D_layers/conv1d4/conv1d/bias/Adam(conv1D_layers/conv1d4/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d5/conv1d/kernel(conv1D_layers/conv1d5/conv1d/kernel/Adam*conv1D_layers/conv1d5/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d5/conv1d/bias&conv1D_layers/conv1d5/conv1d/bias/Adam(conv1D_layers/conv1d5/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
щ
9Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d6/conv1d/kernel(conv1D_layers/conv1d6/conv1d/kernel/Adam*conv1D_layers/conv1d6/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d6/conv1d/bias&conv1D_layers/conv1d6/conv1d/bias/Adam(conv1D_layers/conv1d6/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d7/conv1d/kernel(conv1D_layers/conv1d7/conv1d/kernel/Adam*conv1D_layers/conv1d7/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d7/conv1d/bias&conv1D_layers/conv1d7/conv1d/bias/Adam(conv1D_layers/conv1d7/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d8/conv1d/kernel(conv1D_layers/conv1d8/conv1d/kernel/Adam*conv1D_layers/conv1d8/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d8/conv1d/bias&conv1D_layers/conv1d8/conv1d/bias/Adam(conv1D_layers/conv1d8/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
џ
?Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam	ApplyAdam)classification_layers/dense0/dense/kernel.classification_layers/dense0/dense/kernel/Adam0classification_layers/dense0/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
use_nesterov( *
_output_shapes

: 

Ї
=Adam/update_classification_layers/dense0/dense/bias/ApplyAdam	ApplyAdam'classification_layers/dense0/dense/bias,classification_layers/dense0/dense/bias/Adam.classification_layers/dense0/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
use_nesterov( *
use_locking( 
▓
CAdam/update_classification_layers/dense_last/dense/kernel/ApplyAdam	ApplyAdam-classification_layers/dense_last/dense/kernel2classification_layers/dense_last/dense/kernel/Adam4classification_layers/dense_last/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonWgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
use_nesterov( *
use_locking( 
Ц
AAdam/update_classification_layers/dense_last/dense/bias/ApplyAdam	ApplyAdam+classification_layers/dense_last/dense/bias0classification_layers/dense_last/dense/bias/Adam2classification_layers/dense_last/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonXgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
┬

Adam/mulMulbeta1_power/read
Adam/beta1:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
«
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
validate_shape(*
_output_shapes
: 
─


Adam/mul_1Mulbeta2_power/read
Adam/beta2:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: 
▓
Adam/Assign_1Assignbeta2_power
Adam/mul_1*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
Т	
AdamNoOp:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
о
initNoOp+^conv1D_layers/conv1d1/conv1d/kernel/Assign)^conv1D_layers/conv1d1/conv1d/bias/Assign+^conv1D_layers/conv1d2/conv1d/kernel/Assign)^conv1D_layers/conv1d2/conv1d/bias/Assign+^conv1D_layers/conv1d3/conv1d/kernel/Assign)^conv1D_layers/conv1d3/conv1d/bias/Assign+^conv1D_layers/conv1d4/conv1d/kernel/Assign)^conv1D_layers/conv1d4/conv1d/bias/Assign+^conv1D_layers/conv1d5/conv1d/kernel/Assign)^conv1D_layers/conv1d5/conv1d/bias/Assign+^conv1D_layers/conv1d6/conv1d/kernel/Assign)^conv1D_layers/conv1d6/conv1d/bias/Assign+^conv1D_layers/conv1d7/conv1d/kernel/Assign)^conv1D_layers/conv1d7/conv1d/bias/Assign+^conv1D_layers/conv1d8/conv1d/kernel/Assign)^conv1D_layers/conv1d8/conv1d/bias/Assign1^classification_layers/dense0/dense/kernel/Assign/^classification_layers/dense0/dense/bias/Assign5^classification_layers/dense_last/dense/kernel/Assign3^classification_layers/dense_last/dense/bias/Assign^beta1_power/Assign^beta2_power/Assign0^conv1D_layers/conv1d1/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d1/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d1/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d2/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d2/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d2/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d3/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d3/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d3/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d4/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d4/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d4/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d5/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d5/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d5/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d6/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d6/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d6/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d7/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d7/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d7/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d8/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d8/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d8/conv1d/bias/Adam_1/Assign6^classification_layers/dense0/dense/kernel/Adam/Assign8^classification_layers/dense0/dense/kernel/Adam_1/Assign4^classification_layers/dense0/dense/bias/Adam/Assign6^classification_layers/dense0/dense/bias/Adam_1/Assign:^classification_layers/dense_last/dense/kernel/Adam/Assign<^classification_layers/dense_last/dense/kernel/Adam_1/Assign8^classification_layers/dense_last/dense/bias/Adam/Assign:^classification_layers/dense_last/dense/bias/Adam_1/Assign"ћёєі$     нлЫ	ЏЁ;Ъ^оAJЌ«
Ш(┘(
9
Add
x"T
y"T
z"T"
Ttype:
2	
в
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
╚
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
Ь
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ь
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
A
Equal
x"T
y"T
z
"
Ttype:
2	
љ
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
+
Log
x"T
y"T"
Ttype:	
2


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
┼
MaxPool

input"T
output"T"
Ttype0:
2		"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
в
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	љ
і
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	љ
<
Mul
x"T
y"T
z"T"
Ttype:
2	љ
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
і
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Ѕ
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
s

VariableV2
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ*1.2.12v1.2.0-5-g435cdfcЌФ
|
Input/PlaceholderPlaceholder* 
shape:          *
dtype0*+
_output_shapes
:          
u
Target/PlaceholderPlaceholder*
shape:         *
dtype0*'
_output_shapes
:         
^
conv1D_layers/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
Л
Dconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d1/conv1d/kernel
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Д
*conv1D_layers/conv1d1/conv1d/kernel/AssignAssign#conv1D_layers/conv1d1/conv1d/kernel>conv1D_layers/conv1d1/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d1/conv1d/kernel/readIdentity#conv1D_layers/conv1d1/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d1/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d1/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d1/conv1d/bias/AssignAssign!conv1D_layers/conv1d1/conv1d/bias3conv1D_layers/conv1d1/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d1/conv1d/bias/readIdentity!conv1D_layers/conv1d1/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d1/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d1/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d1/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
М
3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims
ExpandDimsInput/Placeholder7conv1D_layers/conv1d1/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:          *
T0
{
9conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d1/conv1d/kernel/read9conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d1/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:         
╣
0conv1D_layers/conv1d1/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d1/conv1d/convolution/Conv2D*
squeeze_dims
*+
_output_shapes
:         *
T0
о
$conv1D_layers/conv1d1/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d1/conv1d/convolution/Squeeze&conv1D_layers/conv1d1/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d1/conv1d/ReluRelu$conv1D_layers/conv1d1/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d1/dropout/ShapeShape!conv1D_layers/conv1d1/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d1/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
u
0conv1D_layers/conv1d1/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d1/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d1/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d1/dropout/random_uniform/subSub0conv1D_layers/conv1d1/dropout/random_uniform/max0conv1D_layers/conv1d1/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d1/dropout/random_uniform/mulMul:conv1D_layers/conv1d1/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d1/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d1/dropout/random_uniformAdd0conv1D_layers/conv1d1/dropout/random_uniform/mul0conv1D_layers/conv1d1/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d1/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d1/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d1/dropout/FloorFloor!conv1D_layers/conv1d1/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d1/dropout/divRealDiv!conv1D_layers/conv1d1/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d1/dropout/mulMul!conv1D_layers/conv1d1/dropout/div#conv1D_layers/conv1d1/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d2/conv1d/kernel
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Д
*conv1D_layers/conv1d2/conv1d/kernel/AssignAssign#conv1D_layers/conv1d2/conv1d/kernel>conv1D_layers/conv1d2/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d2/conv1d/kernel/readIdentity#conv1D_layers/conv1d2/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d2/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d2/conv1d/bias
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
ј
(conv1D_layers/conv1d2/conv1d/bias/AssignAssign!conv1D_layers/conv1d2/conv1d/bias3conv1D_layers/conv1d2/conv1d/bias/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
░
&conv1D_layers/conv1d2/conv1d/bias/readIdentity!conv1D_layers/conv1d2/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d2/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d2/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d2/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
с
3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d1/dropout/mul7conv1D_layers/conv1d2/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d2/conv1d/kernel/read9conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d2/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d2/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d2/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d2/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d2/conv1d/convolution/Squeeze&conv1D_layers/conv1d2/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d2/conv1d/ReluRelu$conv1D_layers/conv1d2/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d2/dropout/ShapeShape!conv1D_layers/conv1d2/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d2/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d2/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d2/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d2/dropout/random_uniform/subSub0conv1D_layers/conv1d2/dropout/random_uniform/max0conv1D_layers/conv1d2/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d2/dropout/random_uniform/mulMul:conv1D_layers/conv1d2/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d2/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d2/dropout/random_uniformAdd0conv1D_layers/conv1d2/dropout/random_uniform/mul0conv1D_layers/conv1d2/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d2/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d2/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d2/dropout/FloorFloor!conv1D_layers/conv1d2/dropout/add*
_output_shapes
:*
T0
Ї
!conv1D_layers/conv1d2/dropout/divRealDiv!conv1D_layers/conv1d2/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d2/dropout/mulMul!conv1D_layers/conv1d2/dropout/div#conv1D_layers/conv1d2/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
«
Lconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/shape*
T0*"
_output_shapes
:*

seed *6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
dtype0*
seed2 
ф
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/sub*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0
▓
>conv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d3/conv1d/kernel
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Д
*conv1D_layers/conv1d3/conv1d/kernel/AssignAssign#conv1D_layers/conv1d3/conv1d/kernel>conv1D_layers/conv1d3/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
validate_shape(*"
_output_shapes
:
Й
(conv1D_layers/conv1d3/conv1d/kernel/readIdentity#conv1D_layers/conv1d3/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d3/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d3/conv1d/bias
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
ј
(conv1D_layers/conv1d3/conv1d/bias/AssignAssign!conv1D_layers/conv1d3/conv1d/bias3conv1D_layers/conv1d3/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d3/conv1d/bias/readIdentity!conv1D_layers/conv1d3/conv1d/bias*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
T0
Ѓ
.conv1D_layers/conv1d3/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d3/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d3/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d2/dropout/mul7conv1D_layers/conv1d3/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d3/conv1d/kernel/read9conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d3/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d3/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d3/conv1d/convolution/Conv2D*
squeeze_dims
*+
_output_shapes
:         *
T0
о
$conv1D_layers/conv1d3/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d3/conv1d/convolution/Squeeze&conv1D_layers/conv1d3/conv1d/bias/read*+
_output_shapes
:         *
T0*
data_formatNHWC
Ё
!conv1D_layers/conv1d3/conv1d/ReluRelu$conv1D_layers/conv1d3/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d3/dropout/ShapeShape!conv1D_layers/conv1d3/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d3/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d3/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d3/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d3/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d3/dropout/random_uniform/subSub0conv1D_layers/conv1d3/dropout/random_uniform/max0conv1D_layers/conv1d3/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d3/dropout/random_uniform/mulMul:conv1D_layers/conv1d3/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d3/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d3/dropout/random_uniformAdd0conv1D_layers/conv1d3/dropout/random_uniform/mul0conv1D_layers/conv1d3/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d3/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d3/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d3/dropout/FloorFloor!conv1D_layers/conv1d3/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d3/dropout/divRealDiv!conv1D_layers/conv1d3/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d3/dropout/mulMul!conv1D_layers/conv1d3/dropout/div#conv1D_layers/conv1d3/dropout/Floor*
T0*+
_output_shapes
:         
Л
Dconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
«
Lconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d4/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d4/conv1d/kernel/AssignAssign#conv1D_layers/conv1d4/conv1d/kernel>conv1D_layers/conv1d4/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d4/conv1d/kernel/readIdentity#conv1D_layers/conv1d4/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d4/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d4/conv1d/bias
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
ј
(conv1D_layers/conv1d4/conv1d/bias/AssignAssign!conv1D_layers/conv1d4/conv1d/bias3conv1D_layers/conv1d4/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d4/conv1d/bias/readIdentity!conv1D_layers/conv1d4/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d4/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d4/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d4/conv1d/convolution/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
с
3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d3/dropout/mul7conv1D_layers/conv1d4/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d4/conv1d/kernel/read9conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d4/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:         
╣
0conv1D_layers/conv1d4/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d4/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d4/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d4/conv1d/convolution/Squeeze&conv1D_layers/conv1d4/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d4/conv1d/ReluRelu$conv1D_layers/conv1d4/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d4/dropout/ShapeShape!conv1D_layers/conv1d4/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d4/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d4/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d4/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d4/dropout/Shape*

seed *
T0*
dtype0*+
_output_shapes
:         *
seed2 
╝
0conv1D_layers/conv1d4/dropout/random_uniform/subSub0conv1D_layers/conv1d4/dropout/random_uniform/max0conv1D_layers/conv1d4/dropout/random_uniform/min*
_output_shapes
: *
T0
█
0conv1D_layers/conv1d4/dropout/random_uniform/mulMul:conv1D_layers/conv1d4/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d4/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d4/dropout/random_uniformAdd0conv1D_layers/conv1d4/dropout/random_uniform/mul0conv1D_layers/conv1d4/dropout/random_uniform/min*
T0*+
_output_shapes
:         
ћ
!conv1D_layers/conv1d4/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d4/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d4/dropout/FloorFloor!conv1D_layers/conv1d4/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d4/dropout/divRealDiv!conv1D_layers/conv1d4/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d4/dropout/mulMul!conv1D_layers/conv1d4/dropout/div#conv1D_layers/conv1d4/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB"         *
_output_shapes
:*
dtype0
┐
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
valueB
 *   ?*
dtype0*
_output_shapes
: 
«
Lconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:
ф
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/sub*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
▓
>conv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:
О
#conv1D_layers/conv1d5/conv1d/kernel
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Д
*conv1D_layers/conv1d5/conv1d/kernel/AssignAssign#conv1D_layers/conv1d5/conv1d/kernel>conv1D_layers/conv1d5/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d5/conv1d/kernel/readIdentity#conv1D_layers/conv1d5/conv1d/kernel*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
Х
3conv1D_layers/conv1d5/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d5/conv1d/bias
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
dtype0*
	container 
ј
(conv1D_layers/conv1d5/conv1d/bias/AssignAssign!conv1D_layers/conv1d5/conv1d/bias3conv1D_layers/conv1d5/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d5/conv1d/bias/readIdentity!conv1D_layers/conv1d5/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d5/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d5/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d5/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d4/dropout/mul7conv1D_layers/conv1d5/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d5/conv1d/kernel/read9conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d5/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d5/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d5/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d5/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d5/conv1d/convolution/Squeeze&conv1D_layers/conv1d5/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d5/conv1d/ReluRelu$conv1D_layers/conv1d5/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d5/dropout/ShapeShape!conv1D_layers/conv1d5/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d5/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
u
0conv1D_layers/conv1d5/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d5/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d5/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d5/dropout/random_uniform/subSub0conv1D_layers/conv1d5/dropout/random_uniform/max0conv1D_layers/conv1d5/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d5/dropout/random_uniform/mulMul:conv1D_layers/conv1d5/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d5/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d5/dropout/random_uniformAdd0conv1D_layers/conv1d5/dropout/random_uniform/mul0conv1D_layers/conv1d5/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d5/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d5/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d5/dropout/FloorFloor!conv1D_layers/conv1d5/dropout/add*
_output_shapes
:*
T0
Ї
!conv1D_layers/conv1d5/dropout/divRealDiv!conv1D_layers/conv1d5/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d5/dropout/mulMul!conv1D_layers/conv1d5/dropout/div#conv1D_layers/conv1d5/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
valueB
 *   ┐*
_output_shapes
: *
dtype0
┐
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/shape*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
dtype0*
seed2 *

seed 
ф
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d6/conv1d/kernel
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*
dtype0*
	container 
Д
*conv1D_layers/conv1d6/conv1d/kernel/AssignAssign#conv1D_layers/conv1d6/conv1d/kernel>conv1D_layers/conv1d6/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d6/conv1d/kernel/readIdentity#conv1D_layers/conv1d6/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d6/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
├
!conv1D_layers/conv1d6/conv1d/bias
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
ј
(conv1D_layers/conv1d6/conv1d/bias/AssignAssign!conv1D_layers/conv1d6/conv1d/bias3conv1D_layers/conv1d6/conv1d/bias/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
░
&conv1D_layers/conv1d6/conv1d/bias/readIdentity!conv1D_layers/conv1d6/conv1d/bias*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0
Ѓ
.conv1D_layers/conv1d6/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d6/conv1d/convolution/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
y
7conv1D_layers/conv1d6/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d5/dropout/mul7conv1D_layers/conv1d6/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d6/conv1d/kernel/read9conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d6/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:         *
use_cudnn_on_gpu(
╣
0conv1D_layers/conv1d6/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d6/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d6/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d6/conv1d/convolution/Squeeze&conv1D_layers/conv1d6/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d6/conv1d/ReluRelu$conv1D_layers/conv1d6/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d6/dropout/ShapeShape!conv1D_layers/conv1d6/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
u
0conv1D_layers/conv1d6/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d6/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d6/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d6/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d6/dropout/random_uniform/subSub0conv1D_layers/conv1d6/dropout/random_uniform/max0conv1D_layers/conv1d6/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d6/dropout/random_uniform/mulMul:conv1D_layers/conv1d6/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d6/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d6/dropout/random_uniformAdd0conv1D_layers/conv1d6/dropout/random_uniform/mul0conv1D_layers/conv1d6/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d6/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d6/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d6/dropout/FloorFloor!conv1D_layers/conv1d6/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d6/dropout/divRealDiv!conv1D_layers/conv1d6/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d6/dropout/mulMul!conv1D_layers/conv1d6/dropout/div#conv1D_layers/conv1d6/dropout/Floor*
T0*+
_output_shapes
:         
Л
Dconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
_output_shapes
: 
└
Bconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/sub*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0
▓
>conv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d7/conv1d/kernel
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Д
*conv1D_layers/conv1d7/conv1d/kernel/AssignAssign#conv1D_layers/conv1d7/conv1d/kernel>conv1D_layers/conv1d7/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
validate_shape(*"
_output_shapes
:
Й
(conv1D_layers/conv1d7/conv1d/kernel/readIdentity#conv1D_layers/conv1d7/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d7/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d7/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d7/conv1d/bias/AssignAssign!conv1D_layers/conv1d7/conv1d/bias3conv1D_layers/conv1d7/conv1d/bias/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
░
&conv1D_layers/conv1d7/conv1d/bias/readIdentity!conv1D_layers/conv1d7/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d7/conv1d/convolution/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
ђ
6conv1D_layers/conv1d7/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d7/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d6/dropout/mul7conv1D_layers/conv1d7/conv1d/convolution/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
{
9conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
т
5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d7/conv1d/kernel/read9conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1/dim*

Tdim0*
T0*&
_output_shapes
:
и
/conv1D_layers/conv1d7/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
T0*
paddingVALID*/
_output_shapes
:         *
strides
*
data_formatNHWC
╣
0conv1D_layers/conv1d7/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d7/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d7/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d7/conv1d/convolution/Squeeze&conv1D_layers/conv1d7/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d7/conv1d/ReluRelu$conv1D_layers/conv1d7/conv1d/BiasAdd*
T0*+
_output_shapes
:         
ё
#conv1D_layers/conv1d7/dropout/ShapeShape!conv1D_layers/conv1d7/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d7/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d7/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
╠
:conv1D_layers/conv1d7/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d7/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d7/dropout/random_uniform/subSub0conv1D_layers/conv1d7/dropout/random_uniform/max0conv1D_layers/conv1d7/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d7/dropout/random_uniform/mulMul:conv1D_layers/conv1d7/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d7/dropout/random_uniform/sub*
T0*+
_output_shapes
:         
═
,conv1D_layers/conv1d7/dropout/random_uniformAdd0conv1D_layers/conv1d7/dropout/random_uniform/mul0conv1D_layers/conv1d7/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d7/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d7/dropout/random_uniform*
_output_shapes
:*
T0
r
#conv1D_layers/conv1d7/dropout/FloorFloor!conv1D_layers/conv1d7/dropout/add*
T0*
_output_shapes
:
Ї
!conv1D_layers/conv1d7/dropout/divRealDiv!conv1D_layers/conv1d7/conv1d/Reluconv1D_layers/Placeholder*
_output_shapes
:*
T0
д
!conv1D_layers/conv1d7/dropout/mulMul!conv1D_layers/conv1d7/dropout/div#conv1D_layers/conv1d7/dropout/Floor*+
_output_shapes
:         *
T0
Л
Dconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
┐
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
valueB
 *   ┐*
dtype0*
_output_shapes
: 
┐
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
«
Lconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniformDconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/shape*

seed *
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
seed2 *
dtype0*"
_output_shapes
:
ф
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/subSubBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/maxBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
_output_shapes
: *
T0
└
Bconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/mulMulLconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/RandomUniformBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
▓
>conv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniformAddBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/mulBconv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform/min*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0
О
#conv1D_layers/conv1d8/conv1d/kernel
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Д
*conv1D_layers/conv1d8/conv1d/kernel/AssignAssign#conv1D_layers/conv1d8/conv1d/kernel>conv1D_layers/conv1d8/conv1d/kernel/Initializer/random_uniform*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
(conv1D_layers/conv1d8/conv1d/kernel/readIdentity#conv1D_layers/conv1d8/conv1d/kernel*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
Х
3conv1D_layers/conv1d8/conv1d/bias/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
├
!conv1D_layers/conv1d8/conv1d/bias
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
(conv1D_layers/conv1d8/conv1d/bias/AssignAssign!conv1D_layers/conv1d8/conv1d/bias3conv1D_layers/conv1d8/conv1d/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
validate_shape(*
_output_shapes
:
░
&conv1D_layers/conv1d8/conv1d/bias/readIdentity!conv1D_layers/conv1d8/conv1d/bias*
T0*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:
Ѓ
.conv1D_layers/conv1d8/conv1d/convolution/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
ђ
6conv1D_layers/conv1d8/conv1d/convolution/dilation_rateConst*
valueB:*
_output_shapes
:*
dtype0
y
7conv1D_layers/conv1d8/conv1d/convolution/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
с
3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims
ExpandDims!conv1D_layers/conv1d7/dropout/mul7conv1D_layers/conv1d8/conv1d/convolution/ExpandDims/dim*

Tdim0*/
_output_shapes
:         *
T0
{
9conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1/dimConst*
value	B : *
_output_shapes
: *
dtype0
т
5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1
ExpandDims(conv1D_layers/conv1d8/conv1d/kernel/read9conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1/dim*

Tdim0*&
_output_shapes
:*
T0
и
/conv1D_layers/conv1d8/conv1d/convolution/Conv2DConv2D3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1*
paddingVALID*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:         *
use_cudnn_on_gpu(
╣
0conv1D_layers/conv1d8/conv1d/convolution/SqueezeSqueeze/conv1D_layers/conv1d8/conv1d/convolution/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:         
о
$conv1D_layers/conv1d8/conv1d/BiasAddBiasAdd0conv1D_layers/conv1d8/conv1d/convolution/Squeeze&conv1D_layers/conv1d8/conv1d/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:         
Ё
!conv1D_layers/conv1d8/conv1d/ReluRelu$conv1D_layers/conv1d8/conv1d/BiasAdd*+
_output_shapes
:         *
T0
ё
#conv1D_layers/conv1d8/dropout/ShapeShape!conv1D_layers/conv1d8/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
u
0conv1D_layers/conv1d8/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0conv1D_layers/conv1d8/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
╠
:conv1D_layers/conv1d8/dropout/random_uniform/RandomUniformRandomUniform#conv1D_layers/conv1d8/dropout/Shape*+
_output_shapes
:         *
seed2 *
T0*

seed *
dtype0
╝
0conv1D_layers/conv1d8/dropout/random_uniform/subSub0conv1D_layers/conv1d8/dropout/random_uniform/max0conv1D_layers/conv1d8/dropout/random_uniform/min*
T0*
_output_shapes
: 
█
0conv1D_layers/conv1d8/dropout/random_uniform/mulMul:conv1D_layers/conv1d8/dropout/random_uniform/RandomUniform0conv1D_layers/conv1d8/dropout/random_uniform/sub*+
_output_shapes
:         *
T0
═
,conv1D_layers/conv1d8/dropout/random_uniformAdd0conv1D_layers/conv1d8/dropout/random_uniform/mul0conv1D_layers/conv1d8/dropout/random_uniform/min*+
_output_shapes
:         *
T0
ћ
!conv1D_layers/conv1d8/dropout/addAddconv1D_layers/Placeholder,conv1D_layers/conv1d8/dropout/random_uniform*
T0*
_output_shapes
:
r
#conv1D_layers/conv1d8/dropout/FloorFloor!conv1D_layers/conv1d8/dropout/add*
_output_shapes
:*
T0
Ї
!conv1D_layers/conv1d8/dropout/divRealDiv!conv1D_layers/conv1d8/conv1d/Reluconv1D_layers/Placeholder*
T0*
_output_shapes
:
д
!conv1D_layers/conv1d8/dropout/mulMul!conv1D_layers/conv1d8/dropout/div#conv1D_layers/conv1d8/dropout/Floor*+
_output_shapes
:         *
T0
t
2conv1D_layers/conv1d9/max_pooling1d/ExpandDims/dimConst*
value	B :*
_output_shapes
: *
dtype0
┘
.conv1D_layers/conv1d9/max_pooling1d/ExpandDims
ExpandDims!conv1D_layers/conv1d8/dropout/mul2conv1D_layers/conv1d9/max_pooling1d/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:         
з
+conv1D_layers/conv1d9/max_pooling1d/MaxPoolMaxPool.conv1D_layers/conv1d9/max_pooling1d/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:         
░
+conv1D_layers/conv1d9/max_pooling1d/SqueezeSqueeze+conv1D_layers/conv1d9/max_pooling1d/MaxPool*
squeeze_dims
*
T0*+
_output_shapes
:         
x
Flatten/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/Squeeze*
out_type0*
_output_shapes
:*
T0
]
Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
\
Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ђ
Flatten/SliceSliceFlatten/ShapeFlatten/Slice/beginFlatten/Slice/size*
_output_shapes
:*
T0*
Index0
_
Flatten/Slice_1/beginConst*
valueB:*
_output_shapes
:*
dtype0
^
Flatten/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
є
Flatten/Slice_1SliceFlatten/ShapeFlatten/Slice_1/beginFlatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
W
Flatten/ConstConst*
valueB: *
_output_shapes
:*
dtype0
r
Flatten/ProdProdFlatten/Slice_1Flatten/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
Flatten/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0
w
Flatten/ExpandDims
ExpandDimsFlatten/ProdFlatten/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
U
Flatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ї
Flatten/concatConcatV2Flatten/SliceFlatten/ExpandDimsFlatten/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ќ
Flatten/ReshapeReshape+conv1D_layers/conv1d9/max_pooling1d/SqueezeFlatten/concat*
Tshape0*'
_output_shapes
:          *
T0
f
!classification_layers/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
█
Lclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shapeConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB"    
   *
dtype0*
_output_shapes
:
╬
Kclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/meanConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
л
Mclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddevConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
─
Vclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shape*
T0*
_output_shapes

: 
*

seed *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
seed2 
▀
Jclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulMulVclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalMclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 

═
Fclassification_layers/dense0/dense/kernel/Initializer/truncated_normalAddJclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulKclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mean*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
█
)classification_layers/dense0/dense/kernel
VariableV2*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
shape
: 
*
dtype0*
shared_name *
	container 
й
0classification_layers/dense0/dense/kernel/AssignAssign)classification_layers/dense0/dense/kernelFclassification_layers/dense0/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes

: 

╠
.classification_layers/dense0/dense/kernel/readIdentity)classification_layers/dense0/dense/kernel*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
┬
9classification_layers/dense0/dense/bias/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
dtype0*
_output_shapes
:

¤
'classification_layers/dense0/dense/bias
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
dtype0*
	container 
д
.classification_layers/dense0/dense/bias/AssignAssign'classification_layers/dense0/dense/bias9classification_layers/dense0/dense/bias/Initializer/zeros*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
┬
,classification_layers/dense0/dense/bias/readIdentity'classification_layers/dense0/dense/bias*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0
╠
)classification_layers/dense0/dense/MatMulMatMulFlatten/Reshape.classification_layers/dense0/dense/kernel/read*
transpose_b( *'
_output_shapes
:         
*
transpose_a( *
T0
О
*classification_layers/dense0/dense/BiasAddBiasAdd)classification_layers/dense0/dense/MatMul,classification_layers/dense0/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         

Є
!classification_layers/dense0/ReluRelu*classification_layers/dense0/dense/BiasAdd*
T0*'
_output_shapes
:         

І
*classification_layers/dense0/dropout/ShapeShape!classification_layers/dense0/Relu*
out_type0*
_output_shapes
:*
T0
|
7classification_layers/dense0/dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
|
7classification_layers/dense0/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
о
Aclassification_layers/dense0/dropout/random_uniform/RandomUniformRandomUniform*classification_layers/dense0/dropout/Shape*'
_output_shapes
:         
*
seed2 *
T0*

seed *
dtype0
Л
7classification_layers/dense0/dropout/random_uniform/subSub7classification_layers/dense0/dropout/random_uniform/max7classification_layers/dense0/dropout/random_uniform/min*
_output_shapes
: *
T0
В
7classification_layers/dense0/dropout/random_uniform/mulMulAclassification_layers/dense0/dropout/random_uniform/RandomUniform7classification_layers/dense0/dropout/random_uniform/sub*
T0*'
_output_shapes
:         

я
3classification_layers/dense0/dropout/random_uniformAdd7classification_layers/dense0/dropout/random_uniform/mul7classification_layers/dense0/dropout/random_uniform/min*
T0*'
_output_shapes
:         

ф
(classification_layers/dense0/dropout/addAdd!classification_layers/Placeholder3classification_layers/dense0/dropout/random_uniform*
_output_shapes
:*
T0
ђ
*classification_layers/dense0/dropout/FloorFloor(classification_layers/dense0/dropout/add*
_output_shapes
:*
T0
ю
(classification_layers/dense0/dropout/divRealDiv!classification_layers/dense0/Relu!classification_layers/Placeholder*
T0*
_output_shapes
:
и
(classification_layers/dense0/dropout/mulMul(classification_layers/dense0/dropout/div*classification_layers/dense0/dropout/Floor*
T0*'
_output_shapes
:         

с
Pclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shapeConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB"
      *
dtype0*
_output_shapes
:
о
Oclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/meanConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
п
Qclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddevConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
л
Zclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalPclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shape*
T0*
_output_shapes

:
*

seed *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
dtype0*
seed2 
№
Nclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulMulZclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalQclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddev*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

П
Jclassification_layers/dense_last/dense/kernel/Initializer/truncated_normalAddNclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulOclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mean*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

с
-classification_layers/dense_last/dense/kernel
VariableV2*
shared_name *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
	container *
shape
:
*
dtype0*
_output_shapes

:

═
4classification_layers/dense_last/dense/kernel/AssignAssign-classification_layers/dense_last/dense/kernelJclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
validate_shape(*
_output_shapes

:

п
2classification_layers/dense_last/dense/kernel/readIdentity-classification_layers/dense_last/dense/kernel*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

╩
=classification_layers/dense_last/dense/bias/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
О
+classification_layers/dense_last/dense/bias
VariableV2*
	container *
dtype0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
shared_name 
Х
2classification_layers/dense_last/dense/bias/AssignAssign+classification_layers/dense_last/dense/bias=classification_layers/dense_last/dense/bias/Initializer/zeros*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╬
0classification_layers/dense_last/dense/bias/readIdentity+classification_layers/dense_last/dense/bias*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:
ь
-classification_layers/dense_last/dense/MatMulMatMul(classification_layers/dense0/dropout/mul2classification_layers/dense_last/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
с
.classification_layers/dense_last/dense/BiasAddBiasAdd-classification_layers/dense_last/dense/MatMul0classification_layers/dense_last/dense/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
і
classification_layers/SoftmaxSoftmax.classification_layers/dense_last/dense/BiasAdd*
T0*'
_output_shapes
:         
n
)Evaluation_layers/clip_by_value/Minimum/yConst*
valueB
 *  ђ?*
_output_shapes
: *
dtype0
«
'Evaluation_layers/clip_by_value/MinimumMinimumclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*
T0*'
_output_shapes
:         
f
!Evaluation_layers/clip_by_value/yConst*
valueB
 * Т█.*
dtype0*
_output_shapes
: 
е
Evaluation_layers/clip_by_valueMaximum'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:         
o
Evaluation_layers/LogLogEvaluation_layers/clip_by_value*
T0*'
_output_shapes
:         
y
Evaluation_layers/mulMulTarget/PlaceholderEvaluation_layers/Log*
T0*'
_output_shapes
:         
q
'Evaluation_layers/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Д
Evaluation_layers/SumSumEvaluation_layers/mul'Evaluation_layers/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
a
Evaluation_layers/NegNegEvaluation_layers/Sum*
T0*#
_output_shapes
:         
a
Evaluation_layers/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ї
Evaluation_layers/MeanMeanEvaluation_layers/NegEvaluation_layers/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
d
"Evaluation_layers/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
Ъ
Evaluation_layers/ArgMaxArgMaxclassification_layers/Softmax"Evaluation_layers/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:         
f
$Evaluation_layers/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
ў
Evaluation_layers/ArgMax_1ArgMaxTarget/Placeholder$Evaluation_layers/ArgMax_1/dimension*#
_output_shapes
:         *
T0*

Tidx0
ё
Evaluation_layers/EqualEqualEvaluation_layers/ArgMaxEvaluation_layers/ArgMax_1*#
_output_shapes
:         *
T0	
|
Evaluation_layers/accracy/CastCastEvaluation_layers/Equal*

SrcT0
*#
_output_shapes
:         *

DstT0
i
Evaluation_layers/accracy/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ц
Evaluation_layers/accracy/MeanMeanEvaluation_layers/accracy/CastEvaluation_layers/accracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
Evaluation_layers/accuracy/tagsConst*+
value"B  BEvaluation_layers/accuracy*
dtype0*
_output_shapes
: 
Ї
Evaluation_layers/accuracyScalarSummaryEvaluation_layers/accuracy/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
r
Evaluation_layers/loss/tagsConst*'
valueB BEvaluation_layers/loss*
dtype0*
_output_shapes
: 
}
Evaluation_layers/lossScalarSummaryEvaluation_layers/loss/tagsEvaluation_layers/Mean*
T0*
_output_shapes
: 
~
!Evaluation_layers/accuracy_1/tagsConst*-
value$B" BEvaluation_layers/accuracy_1*
_output_shapes
: *
dtype0
Љ
Evaluation_layers/accuracy_1ScalarSummary!Evaluation_layers/accuracy_1/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
T
gradients/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
}
3gradients/Evaluation_layers/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
░
-gradients/Evaluation_layers/Mean_grad/ReshapeReshapegradients/Fill3gradients/Evaluation_layers/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
ђ
+gradients/Evaluation_layers/Mean_grad/ShapeShapeEvaluation_layers/Neg*
T0*
out_type0*
_output_shapes
:
╬
*gradients/Evaluation_layers/Mean_grad/TileTile-gradients/Evaluation_layers/Mean_grad/Reshape+gradients/Evaluation_layers/Mean_grad/Shape*#
_output_shapes
:         *
T0*

Tmultiples0
ѓ
-gradients/Evaluation_layers/Mean_grad/Shape_1ShapeEvaluation_layers/Neg*
out_type0*
_output_shapes
:*
T0
p
-gradients/Evaluation_layers/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
u
+gradients/Evaluation_layers/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
╠
*gradients/Evaluation_layers/Mean_grad/ProdProd-gradients/Evaluation_layers/Mean_grad/Shape_1+gradients/Evaluation_layers/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
w
-gradients/Evaluation_layers/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
л
,gradients/Evaluation_layers/Mean_grad/Prod_1Prod-gradients/Evaluation_layers/Mean_grad/Shape_2-gradients/Evaluation_layers/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
/gradients/Evaluation_layers/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
И
-gradients/Evaluation_layers/Mean_grad/MaximumMaximum,gradients/Evaluation_layers/Mean_grad/Prod_1/gradients/Evaluation_layers/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Х
.gradients/Evaluation_layers/Mean_grad/floordivFloorDiv*gradients/Evaluation_layers/Mean_grad/Prod-gradients/Evaluation_layers/Mean_grad/Maximum*
_output_shapes
: *
T0
њ
*gradients/Evaluation_layers/Mean_grad/CastCast.gradients/Evaluation_layers/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Й
-gradients/Evaluation_layers/Mean_grad/truedivRealDiv*gradients/Evaluation_layers/Mean_grad/Tile*gradients/Evaluation_layers/Mean_grad/Cast*#
_output_shapes
:         *
T0
ї
(gradients/Evaluation_layers/Neg_grad/NegNeg-gradients/Evaluation_layers/Mean_grad/truediv*#
_output_shapes
:         *
T0

*gradients/Evaluation_layers/Sum_grad/ShapeShapeEvaluation_layers/mul*
T0*
out_type0*
_output_shapes
:
k
)gradients/Evaluation_layers/Sum_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
е
(gradients/Evaluation_layers/Sum_grad/addAdd'Evaluation_layers/Sum/reduction_indices)gradients/Evaluation_layers/Sum_grad/Size*
_output_shapes
:*
T0
«
(gradients/Evaluation_layers/Sum_grad/modFloorMod(gradients/Evaluation_layers/Sum_grad/add)gradients/Evaluation_layers/Sum_grad/Size*
_output_shapes
:*
T0
v
,gradients/Evaluation_layers/Sum_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
r
0gradients/Evaluation_layers/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
r
0gradients/Evaluation_layers/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ж
*gradients/Evaluation_layers/Sum_grad/rangeRange0gradients/Evaluation_layers/Sum_grad/range/start)gradients/Evaluation_layers/Sum_grad/Size0gradients/Evaluation_layers/Sum_grad/range/delta*
_output_shapes
:*

Tidx0
q
/gradients/Evaluation_layers/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
х
)gradients/Evaluation_layers/Sum_grad/FillFill,gradients/Evaluation_layers/Sum_grad/Shape_1/gradients/Evaluation_layers/Sum_grad/Fill/value*
T0*
_output_shapes
:
Д
2gradients/Evaluation_layers/Sum_grad/DynamicStitchDynamicStitch*gradients/Evaluation_layers/Sum_grad/range(gradients/Evaluation_layers/Sum_grad/mod*gradients/Evaluation_layers/Sum_grad/Shape)gradients/Evaluation_layers/Sum_grad/Fill*#
_output_shapes
:         *
T0*
N
p
.gradients/Evaluation_layers/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
╔
,gradients/Evaluation_layers/Sum_grad/MaximumMaximum2gradients/Evaluation_layers/Sum_grad/DynamicStitch.gradients/Evaluation_layers/Sum_grad/Maximum/y*#
_output_shapes
:         *
T0
И
-gradients/Evaluation_layers/Sum_grad/floordivFloorDiv*gradients/Evaluation_layers/Sum_grad/Shape,gradients/Evaluation_layers/Sum_grad/Maximum*
_output_shapes
:*
T0
к
,gradients/Evaluation_layers/Sum_grad/ReshapeReshape(gradients/Evaluation_layers/Neg_grad/Neg2gradients/Evaluation_layers/Sum_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
м
)gradients/Evaluation_layers/Sum_grad/TileTile,gradients/Evaluation_layers/Sum_grad/Reshape-gradients/Evaluation_layers/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:         
|
*gradients/Evaluation_layers/mul_grad/ShapeShapeTarget/Placeholder*
out_type0*
_output_shapes
:*
T0
Ђ
,gradients/Evaluation_layers/mul_grad/Shape_1ShapeEvaluation_layers/Log*
out_type0*
_output_shapes
:*
T0
Ж
:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Evaluation_layers/mul_grad/Shape,gradients/Evaluation_layers/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Б
(gradients/Evaluation_layers/mul_grad/mulMul)gradients/Evaluation_layers/Sum_grad/TileEvaluation_layers/Log*
T0*'
_output_shapes
:         
Н
(gradients/Evaluation_layers/mul_grad/SumSum(gradients/Evaluation_layers/mul_grad/mul:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
═
,gradients/Evaluation_layers/mul_grad/ReshapeReshape(gradients/Evaluation_layers/mul_grad/Sum*gradients/Evaluation_layers/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
б
*gradients/Evaluation_layers/mul_grad/mul_1MulTarget/Placeholder)gradients/Evaluation_layers/Sum_grad/Tile*'
_output_shapes
:         *
T0
█
*gradients/Evaluation_layers/mul_grad/Sum_1Sum*gradients/Evaluation_layers/mul_grad/mul_1<gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
М
.gradients/Evaluation_layers/mul_grad/Reshape_1Reshape*gradients/Evaluation_layers/mul_grad/Sum_1,gradients/Evaluation_layers/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
Ю
5gradients/Evaluation_layers/mul_grad/tuple/group_depsNoOp-^gradients/Evaluation_layers/mul_grad/Reshape/^gradients/Evaluation_layers/mul_grad/Reshape_1
б
=gradients/Evaluation_layers/mul_grad/tuple/control_dependencyIdentity,gradients/Evaluation_layers/mul_grad/Reshape6^gradients/Evaluation_layers/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Evaluation_layers/mul_grad/Reshape*'
_output_shapes
:         
е
?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1Identity.gradients/Evaluation_layers/mul_grad/Reshape_16^gradients/Evaluation_layers/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Evaluation_layers/mul_grad/Reshape_1*'
_output_shapes
:         
м
/gradients/Evaluation_layers/Log_grad/Reciprocal
ReciprocalEvaluation_layers/clip_by_value@^gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
М
(gradients/Evaluation_layers/Log_grad/mulMul?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1/gradients/Evaluation_layers/Log_grad/Reciprocal*'
_output_shapes
:         *
T0
Џ
4gradients/Evaluation_layers/clip_by_value_grad/ShapeShape'Evaluation_layers/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:
y
6gradients/Evaluation_layers/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ъ
6gradients/Evaluation_layers/clip_by_value_grad/Shape_2Shape(gradients/Evaluation_layers/Log_grad/mul*
T0*
out_type0*
_output_shapes
:

:gradients/Evaluation_layers/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
4gradients/Evaluation_layers/clip_by_value_grad/zerosFill6gradients/Evaluation_layers/clip_by_value_grad/Shape_2:gradients/Evaluation_layers/clip_by_value_grad/zeros/Const*'
_output_shapes
:         *
T0
╔
;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqualGreaterEqual'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:         
ѕ
Dgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/Evaluation_layers/clip_by_value_grad/Shape6gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ј
5gradients/Evaluation_layers/clip_by_value_grad/SelectSelect;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:         
Г
9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot
LogicalNot;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual*'
_output_shapes
:         
ј
7gradients/Evaluation_layers/clip_by_value_grad/Select_1Select9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:         
Ш
2gradients/Evaluation_layers/clip_by_value_grad/SumSum5gradients/Evaluation_layers/clip_by_value_grad/SelectDgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
в
6gradients/Evaluation_layers/clip_by_value_grad/ReshapeReshape2gradients/Evaluation_layers/clip_by_value_grad/Sum4gradients/Evaluation_layers/clip_by_value_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
Ч
4gradients/Evaluation_layers/clip_by_value_grad/Sum_1Sum7gradients/Evaluation_layers/clip_by_value_grad/Select_1Fgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Я
8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1Reshape4gradients/Evaluation_layers/clip_by_value_grad/Sum_16gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
╗
?gradients/Evaluation_layers/clip_by_value_grad/tuple/group_depsNoOp7^gradients/Evaluation_layers/clip_by_value_grad/Reshape9^gradients/Evaluation_layers/clip_by_value_grad/Reshape_1
╩
Ggradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependencyIdentity6gradients/Evaluation_layers/clip_by_value_grad/Reshape@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape*'
_output_shapes
:         *
T0
┐
Igradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency_1Identity8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape_1*
_output_shapes
: *
T0
Ў
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/ShapeShapeclassification_layers/Softmax*
out_type0*
_output_shapes
:*
T0
Ђ
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
┼
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2ShapeGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Є
Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Щ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zerosFill>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:         *
T0
╔
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual	LessEqualclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*'
_output_shapes
:         *
T0
а
Lgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
┬
=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectSelect@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqualGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:         
║
Agradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNot
LogicalNot@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual*'
_output_shapes
:         
┼
?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1SelectAgradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNotGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:         *
T0
ј
:gradients/Evaluation_layers/clip_by_value/Minimum_grad/SumSum=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectLgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѓ
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeReshape:gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape*
Tshape0*'
_output_shapes
:         *
T0
ћ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1Sum?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1Ngradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1Reshape<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
М
Ggradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_depsNoOp?^gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeA^gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1
Ж
Ogradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeH^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:         *
T0
▀
Qgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1H^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*S
_classI
GEloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: *
T0
┘
0gradients/classification_layers/Softmax_grad/mulMulOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyclassification_layers/Softmax*'
_output_shapes
:         *
T0
ї
Bgradients/classification_layers/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Э
0gradients/classification_layers/Softmax_grad/SumSum0gradients/classification_layers/Softmax_grad/mulBgradients/classification_layers/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
І
:gradients/classification_layers/Softmax_grad/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ь
4gradients/classification_layers/Softmax_grad/ReshapeReshape0gradients/classification_layers/Softmax_grad/Sum:gradients/classification_layers/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
­
0gradients/classification_layers/Softmax_grad/subSubOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency4gradients/classification_layers/Softmax_grad/Reshape*'
_output_shapes
:         *
T0
╝
2gradients/classification_layers/Softmax_grad/mul_1Mul0gradients/classification_layers/Softmax_grad/subclassification_layers/Softmax*'
_output_shapes
:         *
T0
╚
Igradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/classification_layers/Softmax_grad/mul_1*
_output_shapes
:*
T0*
data_formatNHWC
О
Ngradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/classification_layers/Softmax_grad/mul_1J^gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad
Я
Vgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/classification_layers/Softmax_grad/mul_1O^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/classification_layers/Softmax_grad/mul_1*'
_output_shapes
:         
Ѓ
Xgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradO^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
▒
Cgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulMatMulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency2classification_layers/dense_last/dense/kernel/read*
transpose_b(*
T0*'
_output_shapes
:         
*
transpose_a( 
а
Egradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1MatMul(classification_layers/dense0/dropout/mulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
с
Mgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_depsNoOpD^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMulF^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1
ђ
Ugradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependencyIdentityCgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulN^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul*'
_output_shapes
:         
*
T0
§
Wgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1N^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*X
_classN
LJloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1*
_output_shapes

:
*
T0
«
=gradients/classification_layers/dense0/dropout/mul_grad/ShapeShape(classification_layers/dense0/dropout/div*
T0*
out_type0*#
_output_shapes
:         
▓
?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1Shape*classification_layers/dense0/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
Б
Mgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/mul_grad/Shape?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
У
;gradients/classification_layers/dense0/dropout/mul_grad/mulMulUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*classification_layers/dense0/dropout/Floor*
_output_shapes
:*
T0
ј
;gradients/classification_layers/dense0/dropout/mul_grad/SumSum;gradients/classification_layers/dense0/dropout/mul_grad/mulMgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
э
?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/mul_grad/Sum=gradients/classification_layers/dense0/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
У
=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Mul(classification_layers/dense0/dropout/divUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
ћ
=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1Sum=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Ogradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
§
Agradients/classification_layers/dense0/dropout/mul_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
о
Hgradients/classification_layers/dense0/dropout/mul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/mul_grad/ReshapeB^gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1
▀
Pgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeI^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape*
_output_shapes
:
т
Rgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/mul_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
ъ
=gradients/classification_layers/dense0/dropout/div_grad/ShapeShape!classification_layers/dense0/Relu*
T0*
out_type0*
_output_shapes
:
Е
?gradients/classification_layers/dense0/dropout/div_grad/Shape_1Shape!classification_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
Б
Mgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/div_grad/Shape?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Р
?gradients/classification_layers/dense0/dropout/div_grad/RealDivRealDivPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency!classification_layers/Placeholder*
T0*
_output_shapes
:
њ
;gradients/classification_layers/dense0/dropout/div_grad/SumSum?gradients/classification_layers/dense0/dropout/div_grad/RealDivMgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
є
?gradients/classification_layers/dense0/dropout/div_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/div_grad/Sum=gradients/classification_layers/dense0/dropout/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

Ќ
;gradients/classification_layers/dense0/dropout/div_grad/NegNeg!classification_layers/dense0/Relu*
T0*'
_output_shapes
:         

¤
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_1RealDiv;gradients/classification_layers/dense0/dropout/div_grad/Neg!classification_layers/Placeholder*
_output_shapes
:*
T0
Н
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_2RealDivAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_1!classification_layers/Placeholder*
T0*
_output_shapes
:
Щ
;gradients/classification_layers/dense0/dropout/div_grad/mulMulPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
њ
=gradients/classification_layers/dense0/dropout/div_grad/Sum_1Sum;gradients/classification_layers/dense0/dropout/div_grad/mulOgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
§
Agradients/classification_layers/dense0/dropout/div_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/div_grad/Sum_1?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
о
Hgradients/classification_layers/dense0/dropout/div_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/div_grad/ReshapeB^gradients/classification_layers/dense0/dropout/div_grad/Reshape_1
Ь
Pgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/div_grad/ReshapeI^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape*'
_output_shapes
:         

т
Rgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/div_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape_1*
_output_shapes
:
В
9gradients/classification_layers/dense0/Relu_grad/ReluGradReluGradPgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency!classification_layers/dense0/Relu*
T0*'
_output_shapes
:         

╦
Egradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/classification_layers/dense0/Relu_grad/ReluGrad*
_output_shapes
:
*
T0*
data_formatNHWC
о
Jgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_depsNoOp:^gradients/classification_layers/dense0/Relu_grad/ReluGradF^gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad
Т
Rgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/classification_layers/dense0/Relu_grad/ReluGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/classification_layers/dense0/Relu_grad/ReluGrad*'
_output_shapes
:         

з
Tgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Ц
?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulMatMulRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency.classification_layers/dense0/dense/kernel/read*
transpose_b(*'
_output_shapes
:          *
transpose_a( *
T0
 
Agradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1MatMulFlatten/ReshapeRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

: 
*
transpose_a(
О
Igradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dense/MatMul_grad/MatMulB^gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1
­
Qgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulJ^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul*'
_output_shapes
:          *
T0
ь
Sgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1J^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1*
_output_shapes

: 

Ј
$gradients/Flatten/Reshape_grad/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/Squeeze*
out_type0*
_output_shapes
:*
T0
Ь
&gradients/Flatten/Reshape_grad/ReshapeReshapeQgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency$gradients/Flatten/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ф
@gradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/ShapeShape+conv1D_layers/conv1d9/max_pooling1d/MaxPool*
out_type0*
_output_shapes
:*
T0
 
Bgradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/ReshapeReshape&gradients/Flatten/Reshape_grad/Reshape@gradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
Ѓ
Fgradients/conv1D_layers/conv1d9/max_pooling1d/MaxPool_grad/MaxPoolGradMaxPoolGrad.conv1D_layers/conv1d9/max_pooling1d/ExpandDims+conv1D_layers/conv1d9/max_pooling1d/MaxPoolBgradients/conv1D_layers/conv1d9/max_pooling1d/Squeeze_grad/Reshape*
ksize
*
T0*
paddingVALID*/
_output_shapes
:         *
data_formatNHWC*
strides

ц
Cgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d8/dropout/mul*
out_type0*
_output_shapes
:*
T0
А
Egradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/ReshapeReshapeFgradients/conv1D_layers/conv1d9/max_pooling1d/MaxPool_grad/MaxPoolGradCgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
а
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d8/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d8/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╩
4gradients/conv1D_layers/conv1d8/dropout/mul_grad/mulMulEgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Reshape#conv1D_layers/conv1d8/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d8/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d8/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d8/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
╩
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d8/dropout/divEgradients/conv1D_layers/conv1d9/max_pooling1d/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d8/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d8/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d8/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d8/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d8/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d8/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d8/dropout/div_grad/ShapeShape!conv1D_layers/conv1d8/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d8/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ш
8gradients/conv1D_layers/conv1d8/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d8/dropout/div_grad/NegNeg!conv1D_layers/conv1d8/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d8/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d8/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d8/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d8/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d8/dropout/div_grad/mulHgradients/conv1D_layers/conv1d8/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d8/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d8/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d8/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d8/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d8/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d8/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d8/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d8/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d8/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d8/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d8/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d7/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d8/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d7/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d7/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d7/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d7/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d7/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d7/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d7/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d7/dropout/divJgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d7/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d7/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d7/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d7/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d7/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d7/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d7/dropout/div_grad/ShapeShape!conv1D_layers/conv1d7/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d7/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d7/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d7/dropout/div_grad/NegNeg!conv1D_layers/conv1d7/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d7/dropout/div_grad/Negconv1D_layers/Placeholder*
_output_shapes
:*
T0
┐
:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d7/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d7/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d7/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d7/dropout/div_grad/mulHgradients/conv1D_layers/conv1d7/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d7/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d7/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d7/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d7/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d7/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d7/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d7/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d7/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d7/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d7/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d7/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d6/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d7/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d6/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d6/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d6/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d6/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d6/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d6/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d6/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d6/dropout/divJgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d6/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d6/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d6/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d6/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d6/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d6/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d6/dropout/div_grad/ShapeShape!conv1D_layers/conv1d6/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d6/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ш
8gradients/conv1D_layers/conv1d6/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d6/dropout/div_grad/NegNeg!conv1D_layers/conv1d6/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d6/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d6/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d6/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d6/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d6/dropout/div_grad/mulHgradients/conv1D_layers/conv1d6/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d6/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d6/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d6/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape*+
_output_shapes
:         *
T0
╔
Kgradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d6/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d6/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d6/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d6/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d6/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d6/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Reshape*J
_output_shapes8
6:4                                    *
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
Ъ
Fgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d6/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d6/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
ѓ
Ogradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d5/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d6/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d5/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d5/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d5/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d5/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d5/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d5/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d5/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d5/dropout/divJgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d5/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d5/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d5/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d5/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d5/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d5/dropout/mul_grad/Reshape_1*
_output_shapes
:
Ќ
6gradients/conv1D_layers/conv1d5/dropout/div_grad/ShapeShape!conv1D_layers/conv1d5/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d5/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d5/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d5/dropout/div_grad/NegNeg!conv1D_layers/conv1d5/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d5/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d5/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d5/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d5/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d5/dropout/div_grad/mulHgradients/conv1D_layers/conv1d5/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d5/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d5/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d5/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d5/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d5/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d5/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d5/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d5/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d5/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4                                    *
use_cudnn_on_gpu(
Ъ
Fgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d5/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d5/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d4/dropout/mul*
out_type0*
_output_shapes
:*
T0
╝
Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d5/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/Shape*
Tshape0*"
_output_shapes
:*
T0
а
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d4/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d4/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d4/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d4/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d4/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d4/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d4/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d4/dropout/divJgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d4/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d4/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d4/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d4/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d4/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape*
_output_shapes
:
╔
Kgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d4/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d4/dropout/div_grad/ShapeShape!conv1D_layers/conv1d4/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d4/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d4/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d4/dropout/div_grad/NegNeg!conv1D_layers/conv1d4/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d4/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d4/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d4/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d4/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
6gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d4/dropout/div_grad/mulHgradients/conv1D_layers/conv1d4/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d4/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d4/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d4/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d4/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d4/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d4/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d4/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d4/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d4/conv1d/convolution/ExpandDims*
out_type0*
_output_shapes
:*
T0
▄
Rgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4                                    
Ъ
Fgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d4/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d4/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:*
data_formatNHWC*
strides

ѓ
Ogradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         
Ц
Ygradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d3/dropout/mul*
T0*
out_type0*
_output_shapes
:
╝
Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
_output_shapes
:*
dtype0
╣
Lgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d4/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d3/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d3/dropout/Floor*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
¤
4gradients/conv1D_layers/conv1d3/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d3/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d3/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d3/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d3/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d3/dropout/divJgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d3/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d3/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d3/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d3/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d3/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d3/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
Ќ
6gradients/conv1D_layers/conv1d3/dropout/div_grad/ShapeShape!conv1D_layers/conv1d3/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d3/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d3/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d3/dropout/div_grad/NegNeg!conv1D_layers/conv1d3/conv1d/Relu*+
_output_shapes
:         *
T0
╣
:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d3/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
_output_shapes
:*
T0
т
4gradients/conv1D_layers/conv1d3/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d3/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d3/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d3/dropout/div_grad/mulHgradients/conv1D_layers/conv1d3/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d3/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d3/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d3/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d3/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d3/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d3/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/conv1D_layers/conv1d3/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         
█
Ngradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d3/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d3/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4                                    *
data_formatNHWC*
strides

Ъ
Fgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
║
Sgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d3/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d3/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*&
_output_shapes
:*
data_formatNHWC*
strides

ѓ
Ogradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Е
Hgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d2/dropout/mul*
out_type0*
_output_shapes
:*
T0
╝
Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Ъ
Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d3/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d2/dropout/div*
T0*
out_type0*#
_output_shapes
:         
ц
8gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d2/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d2/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d2/dropout/Floor*
_output_shapes
:*
T0
щ
4gradients/conv1D_layers/conv1d2/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d2/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Р
8gradients/conv1D_layers/conv1d2/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¤
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d2/dropout/divJgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_grad/Reshape*
_output_shapes
:*
T0
 
6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d2/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d2/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d2/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d2/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d2/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d2/dropout/mul_grad/Reshape_1*
_output_shapes
:
Ќ
6gradients/conv1D_layers/conv1d2/dropout/div_grad/ShapeShape!conv1D_layers/conv1d2/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
џ
8gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
out_type0*#
_output_shapes
:         *
T0
ј
Fgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╠
8gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
_output_shapes
:*
T0
§
4gradients/conv1D_layers/conv1d2/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ш
8gradients/conv1D_layers/conv1d2/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
ћ
4gradients/conv1D_layers/conv1d2/dropout/div_grad/NegNeg!conv1D_layers/conv1d2/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d2/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d2/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d2/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d2/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d2/dropout/div_grad/mulHgradients/conv1D_layers/conv1d2/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d2/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d2/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d2/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d2/dropout/div_grad/Reshape_1*
_output_shapes
:
ж
9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d2/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d2/conv1d/Relu*+
_output_shapes
:         *
T0
┼
?gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
╩
Dgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d2/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
┤
Egradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d2/conv1d/convolution/Conv2D*
out_type0*
_output_shapes
:*
T0
»
Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Shape*
Tshape0*/
_output_shapes
:         *
T0
и
Dgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d2/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4                                    *
data_formatNHWC*
strides

Ъ
Fgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d2/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d2/conv1d/convolution/Squeeze_grad/Reshape*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingVALID
ѓ
Ogradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:         *
T0
Ц
Ygradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
Е
Hgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/ShapeShape!conv1D_layers/conv1d1/dropout/mul*
out_type0*
_output_shapes
:*
T0
╝
Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/ReshapeReshapeWgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependencyHgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
Ъ
Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d2/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/ShapeShape!conv1D_layers/conv1d1/dropout/div*
out_type0*#
_output_shapes
:         *
T0
ц
8gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1Shape#conv1D_layers/conv1d1/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape8gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
¤
4gradients/conv1D_layers/conv1d1/dropout/mul_grad/mulMulJgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Reshape#conv1D_layers/conv1d1/dropout/Floor*
T0*
_output_shapes
:
щ
4gradients/conv1D_layers/conv1d1/dropout/mul_grad/SumSum4gradients/conv1D_layers/conv1d1/dropout/mul_grad/mulFgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
8gradients/conv1D_layers/conv1d1/dropout/mul_grad/ReshapeReshape4gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0
¤
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/mul_1Mul!conv1D_layers/conv1d1/dropout/divJgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_grad/Reshape*
T0*
_output_shapes
:
 
6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum_1Sum6gradients/conv1D_layers/conv1d1/dropout/mul_grad/mul_1Hgradients/conv1D_layers/conv1d1/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
У
:gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d1/dropout/mul_grad/Sum_18gradients/conv1D_layers/conv1d1/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
┴
Agradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape;^gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1
├
Igradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d1/dropout/mul_grad/ReshapeB^gradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
╔
Kgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1B^gradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1D_layers/conv1d1/dropout/mul_grad/Reshape_1*
_output_shapes
:
Ќ
6gradients/conv1D_layers/conv1d1/dropout/div_grad/ShapeShape!conv1D_layers/conv1d1/conv1d/Relu*
out_type0*
_output_shapes
:*
T0
џ
8gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1Shapeconv1D_layers/Placeholder*
T0*
out_type0*#
_output_shapes
:         
ј
Fgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape8gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
8gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDivRealDivIgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependencyconv1D_layers/Placeholder*
T0*
_output_shapes
:
§
4gradients/conv1D_layers/conv1d1/dropout/div_grad/SumSum8gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDivFgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
8gradients/conv1D_layers/conv1d1/dropout/div_grad/ReshapeReshape4gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum6gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape*
Tshape0*+
_output_shapes
:         *
T0
ћ
4gradients/conv1D_layers/conv1d1/dropout/div_grad/NegNeg!conv1D_layers/conv1d1/conv1d/Relu*
T0*+
_output_shapes
:         
╣
:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_1RealDiv4gradients/conv1D_layers/conv1d1/dropout/div_grad/Negconv1D_layers/Placeholder*
T0*
_output_shapes
:
┐
:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_2RealDiv:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_1conv1D_layers/Placeholder*
T0*
_output_shapes
:
т
4gradients/conv1D_layers/conv1d1/dropout/div_grad/mulMulIgradients/conv1D_layers/conv1d1/dropout/mul_grad/tuple/control_dependency:gradients/conv1D_layers/conv1d1/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
6gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum_1Sum4gradients/conv1D_layers/conv1d1/dropout/div_grad/mulHgradients/conv1D_layers/conv1d1/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
:gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1Reshape6gradients/conv1D_layers/conv1d1/dropout/div_grad/Sum_18gradients/conv1D_layers/conv1d1/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
┴
Agradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_depsNoOp9^gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape;^gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1
о
Igradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependencyIdentity8gradients/conv1D_layers/conv1d1/dropout/div_grad/ReshapeB^gradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape*+
_output_shapes
:         
╔
Kgradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependency_1Identity:gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1B^gradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/group_deps*M
_classC
A?loc:@gradients/conv1D_layers/conv1d1/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
ж
9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGradReluGradIgradients/conv1D_layers/conv1d1/dropout/div_grad/tuple/control_dependency!conv1D_layers/conv1d1/conv1d/Relu*
T0*+
_output_shapes
:         
┼
?gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
╩
Dgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_depsNoOp:^gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad@^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGrad
я
Lgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGradE^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients/conv1D_layers/conv1d1/conv1d/Relu_grad/ReluGrad*+
_output_shapes
:         *
T0
█
Ngradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGradE^gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/group_deps*R
_classH
FDloc:@gradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
┤
Egradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/ShapeShape/conv1D_layers/conv1d1/conv1d/convolution/Conv2D*
T0*
out_type0*
_output_shapes
:
»
Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/ReshapeReshapeLgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependencyEgradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         
и
Dgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/ShapeShape3conv1D_layers/conv1d1/conv1d/convolution/ExpandDims*
T0*
out_type0*
_output_shapes
:
▄
Rgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputDgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape5conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
T0*
paddingVALID*J
_output_shapes8
6:4                                    *
data_formatNHWC*
strides

Ъ
Fgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
║
Sgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter3conv1D_layers/conv1d1/conv1d/convolution/ExpandDimsFgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Shape_1Ggradients/conv1D_layers/conv1d1/conv1d/convolution/Squeeze_grad/Reshape*
paddingVALID*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:*
use_cudnn_on_gpu(
ѓ
Ogradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_depsNoOpS^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputT^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter
ф
Wgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependencyIdentityRgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInputP^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:          *
T0
Ц
Ygradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1IdentitySgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilterP^gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
Ъ
Jgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
╣
Lgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/ReshapeReshapeYgradients/conv1D_layers/conv1d1/conv1d/convolution/Conv2D_grad/tuple/control_dependency_1Jgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
ќ
beta1_power/initial_valueConst*
valueB
 *fff?*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
dtype0
Д
beta1_power
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
к
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
ѓ
beta1_power/readIdentitybeta1_power*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
ќ
beta2_power/initial_valueConst*
valueB
 *wЙ?*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
dtype0
Д
beta2_power
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
к
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
ѓ
beta2_power/readIdentitybeta2_power*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0
¤
:conv1D_layers/conv1d1/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d1/conv1d/kernel/Adam
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Г
/conv1D_layers/conv1d1/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d1/conv1d/kernel/Adam:conv1D_layers/conv1d1/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d1/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d1/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d1/conv1d/kernel/Adam_1
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
│
1conv1D_layers/conv1d1/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d1/conv1d/kernel/Adam_1<conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d1/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d1/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d1/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d1/conv1d/bias/Adam
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Ю
-conv1D_layers/conv1d1/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d1/conv1d/bias/Adam8conv1D_layers/conv1d1/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d1/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d1/conv1d/bias/Adam*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:
й
:conv1D_layers/conv1d1/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d1/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d1/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d1/conv1d/bias/Adam_1:conv1D_layers/conv1d1/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d1/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d1/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d2/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d2/conv1d/kernel/Adam
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
Г
/conv1D_layers/conv1d2/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d2/conv1d/kernel/Adam:conv1D_layers/conv1d2/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d2/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d2/conv1d/kernel/Adam*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0
Л
<conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d2/conv1d/kernel/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
│
1conv1D_layers/conv1d2/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d2/conv1d/kernel/Adam_1<conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d2/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d2/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d2/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d2/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d2/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d2/conv1d/bias/Adam8conv1D_layers/conv1d2/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d2/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d2/conv1d/bias/Adam*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:
й
:conv1D_layers/conv1d2/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d2/conv1d/bias/Adam_1
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Б
/conv1D_layers/conv1d2/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d2/conv1d/bias/Adam_1:conv1D_layers/conv1d2/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d2/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d2/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
_output_shapes
:*
T0
¤
:conv1D_layers/conv1d3/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d3/conv1d/kernel/Adam
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
dtype0*
	container 
Г
/conv1D_layers/conv1d3/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d3/conv1d/kernel/Adam:conv1D_layers/conv1d3/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d3/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d3/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d3/conv1d/kernel/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
│
1conv1D_layers/conv1d3/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d3/conv1d/kernel/Adam_1<conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d3/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d3/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d3/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d3/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d3/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d3/conv1d/bias/Adam8conv1D_layers/conv1d3/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d3/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d3/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d3/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d3/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d3/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d3/conv1d/bias/Adam_1:conv1D_layers/conv1d3/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d3/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d3/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d4/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d4/conv1d/kernel/Adam
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Г
/conv1D_layers/conv1d4/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d4/conv1d/kernel/Adam:conv1D_layers/conv1d4/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d4/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d4/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d4/conv1d/kernel/Adam_1
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
│
1conv1D_layers/conv1d4/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d4/conv1d/kernel/Adam_1<conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d4/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d4/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d4/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d4/conv1d/bias/Adam
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Ю
-conv1D_layers/conv1d4/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d4/conv1d/bias/Adam8conv1D_layers/conv1d4/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d4/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d4/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d4/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d4/conv1d/bias/Adam_1
VariableV2*
	container *
dtype0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
shape:*
shared_name 
Б
/conv1D_layers/conv1d4/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d4/conv1d/bias/Adam_1:conv1D_layers/conv1d4/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d4/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d4/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
_output_shapes
:*
T0
¤
:conv1D_layers/conv1d5/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d5/conv1d/kernel/Adam
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
dtype0*
	container 
Г
/conv1D_layers/conv1d5/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d5/conv1d/kernel/Adam:conv1D_layers/conv1d5/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d5/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d5/conv1d/kernel/Adam*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
Л
<conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d5/conv1d/kernel/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
│
1conv1D_layers/conv1d5/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d5/conv1d/kernel/Adam_1<conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d5/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d5/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d5/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d5/conv1d/bias/Adam
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
-conv1D_layers/conv1d5/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d5/conv1d/bias/Adam8conv1D_layers/conv1d5/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d5/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d5/conv1d/bias/Adam*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:
й
:conv1D_layers/conv1d5/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d5/conv1d/bias/Adam_1
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Б
/conv1D_layers/conv1d5/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d5/conv1d/bias/Adam_1:conv1D_layers/conv1d5/conv1d/bias/Adam_1/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Й
-conv1D_layers/conv1d5/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d5/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d6/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d6/conv1d/kernel/Adam
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
Г
/conv1D_layers/conv1d6/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d6/conv1d/kernel/Adam:conv1D_layers/conv1d6/conv1d/kernel/Adam/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╚
-conv1D_layers/conv1d6/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d6/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d6/conv1d/kernel/Adam_1
VariableV2*
	container *
dtype0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
shape:*
shared_name 
│
1conv1D_layers/conv1d6/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d6/conv1d/kernel/Adam_1<conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d6/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d6/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d6/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d6/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d6/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d6/conv1d/bias/Adam8conv1D_layers/conv1d6/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d6/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d6/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d6/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╩
(conv1D_layers/conv1d6/conv1d/bias/Adam_1
VariableV2*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Б
/conv1D_layers/conv1d6/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d6/conv1d/bias/Adam_1:conv1D_layers/conv1d6/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d6/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d6/conv1d/bias/Adam_1*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0
¤
:conv1D_layers/conv1d7/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
▄
(conv1D_layers/conv1d7/conv1d/kernel/Adam
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Г
/conv1D_layers/conv1d7/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d7/conv1d/kernel/Adam:conv1D_layers/conv1d7/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d7/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d7/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
я
*conv1D_layers/conv1d7/conv1d/kernel/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
	container *
shape:*
dtype0*"
_output_shapes
:
│
1conv1D_layers/conv1d7/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d7/conv1d/kernel/Adam_1<conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Initializer/zeros*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0*
validate_shape(*
use_locking(
╠
/conv1D_layers/conv1d7/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d7/conv1d/kernel/Adam_1*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*"
_output_shapes
:*
T0
╗
8conv1D_layers/conv1d7/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
dtype0*
_output_shapes
:
╚
&conv1D_layers/conv1d7/conv1d/bias/Adam
VariableV2*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ю
-conv1D_layers/conv1d7/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d7/conv1d/bias/Adam8conv1D_layers/conv1d7/conv1d/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
validate_shape(*
_output_shapes
:
║
+conv1D_layers/conv1d7/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d7/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d7/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d7/conv1d/bias/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
dtype0*
	container 
Б
/conv1D_layers/conv1d7/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d7/conv1d/bias/Adam_1:conv1D_layers/conv1d7/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d7/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d7/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
_output_shapes
:
¤
:conv1D_layers/conv1d8/conv1d/kernel/Adam/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB*    *"
_output_shapes
:*
dtype0
▄
(conv1D_layers/conv1d8/conv1d/kernel/Adam
VariableV2*
shape:*"
_output_shapes
:*
shared_name *6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
dtype0*
	container 
Г
/conv1D_layers/conv1d8/conv1d/kernel/Adam/AssignAssign(conv1D_layers/conv1d8/conv1d/kernel/Adam:conv1D_layers/conv1d8/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╚
-conv1D_layers/conv1d8/conv1d/kernel/Adam/readIdentity(conv1D_layers/conv1d8/conv1d/kernel/Adam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
Л
<conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Initializer/zerosConst*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*!
valueB*    *
dtype0*"
_output_shapes
:
я
*conv1D_layers/conv1d8/conv1d/kernel/Adam_1
VariableV2*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
│
1conv1D_layers/conv1d8/conv1d/kernel/Adam_1/AssignAssign*conv1D_layers/conv1d8/conv1d/kernel/Adam_1<conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*
validate_shape(*"
_output_shapes
:
╠
/conv1D_layers/conv1d8/conv1d/kernel/Adam_1/readIdentity*conv1D_layers/conv1d8/conv1d/kernel/Adam_1*
T0*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:
╗
8conv1D_layers/conv1d8/conv1d/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╚
&conv1D_layers/conv1d8/conv1d/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
dtype0*
	container 
Ю
-conv1D_layers/conv1d8/conv1d/bias/Adam/AssignAssign&conv1D_layers/conv1d8/conv1d/bias/Adam8conv1D_layers/conv1d8/conv1d/bias/Adam/Initializer/zeros*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
║
+conv1D_layers/conv1d8/conv1d/bias/Adam/readIdentity&conv1D_layers/conv1d8/conv1d/bias/Adam*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0
й
:conv1D_layers/conv1d8/conv1d/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
valueB*    *
_output_shapes
:*
dtype0
╩
(conv1D_layers/conv1d8/conv1d/bias/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
dtype0*
	container 
Б
/conv1D_layers/conv1d8/conv1d/bias/Adam_1/AssignAssign(conv1D_layers/conv1d8/conv1d/bias/Adam_1:conv1D_layers/conv1d8/conv1d/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
validate_shape(*
_output_shapes
:
Й
-conv1D_layers/conv1d8/conv1d/bias/Adam_1/readIdentity(conv1D_layers/conv1d8/conv1d/bias/Adam_1*
T0*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:
М
@classification_layers/dense0/dense/kernel/Adam/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB 
*    *
_output_shapes

: 
*
dtype0
Я
.classification_layers/dense0/dense/kernel/Adam
VariableV2*
shape
: 
*
_output_shapes

: 
*
shared_name *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
	container 
┴
5classification_layers/dense0/dense/kernel/Adam/AssignAssign.classification_layers/dense0/dense/kernel/Adam@classification_layers/dense0/dense/kernel/Adam/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
validate_shape(*
use_locking(
о
3classification_layers/dense0/dense/kernel/Adam/readIdentity.classification_layers/dense0/dense/kernel/Adam*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
Н
Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB 
*    *
_output_shapes

: 
*
dtype0
Р
0classification_layers/dense0/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
shape
: 
*
shared_name 
К
7classification_layers/dense0/dense/kernel/Adam_1/AssignAssign0classification_layers/dense0/dense/kernel/Adam_1Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
validate_shape(*
use_locking(
┌
5classification_layers/dense0/dense/kernel/Adam_1/readIdentity0classification_layers/dense0/dense/kernel/Adam_1*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0
К
>classification_layers/dense0/dense/bias/Adam/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
dtype0*
_output_shapes
:

н
,classification_layers/dense0/dense/bias/Adam
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
dtype0*
	container 
х
3classification_layers/dense0/dense/bias/Adam/AssignAssign,classification_layers/dense0/dense/bias/Adam>classification_layers/dense0/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
validate_shape(*
_output_shapes
:

╠
1classification_layers/dense0/dense/bias/Adam/readIdentity,classification_layers/dense0/dense/bias/Adam*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:

╔
@classification_layers/dense0/dense/bias/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
dtype0*
_output_shapes
:

о
.classification_layers/dense0/dense/bias/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
dtype0*
	container 
╗
5classification_layers/dense0/dense/bias/Adam_1/AssignAssign.classification_layers/dense0/dense/bias/Adam_1@classification_layers/dense0/dense/bias/Adam_1/Initializer/zeros*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
л
3classification_layers/dense0/dense/bias/Adam_1/readIdentity.classification_layers/dense0/dense/bias/Adam_1*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:

█
Dclassification_layers/dense_last/dense/kernel/Adam/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

У
2classification_layers/dense_last/dense/kernel/Adam
VariableV2*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
shape
:
*
dtype0*
shared_name *
	container 
Л
9classification_layers/dense_last/dense/kernel/Adam/AssignAssign2classification_layers/dense_last/dense/kernel/AdamDclassification_layers/dense_last/dense/kernel/Adam/Initializer/zeros*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
Р
7classification_layers/dense_last/dense/kernel/Adam/readIdentity2classification_layers/dense_last/dense/kernel/Adam*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

П
Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
_output_shapes

:
*
dtype0
Ж
4classification_layers/dense_last/dense/kernel/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
shared_name *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
dtype0*
	container 
О
;classification_layers/dense_last/dense/kernel/Adam_1/AssignAssign4classification_layers/dense_last/dense/kernel/Adam_1Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zeros*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
Т
9classification_layers/dense_last/dense/kernel/Adam_1/readIdentity4classification_layers/dense_last/dense/kernel/Adam_1*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
¤
Bclassification_layers/dense_last/dense/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
▄
0classification_layers/dense_last/dense/bias/Adam
VariableV2*
	container *
dtype0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
shared_name 
┼
7classification_layers/dense_last/dense/bias/Adam/AssignAssign0classification_layers/dense_last/dense/bias/AdamBclassification_layers/dense_last/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
validate_shape(*
_output_shapes
:
п
5classification_layers/dense_last/dense/bias/Adam/readIdentity0classification_layers/dense_last/dense/bias/Adam*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0
Л
Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
я
2classification_layers/dense_last/dense/bias/Adam_1
VariableV2*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
╦
9classification_layers/dense_last/dense/bias/Adam_1/AssignAssign2classification_layers/dense_last/dense/bias/Adam_1Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
validate_shape(*
_output_shapes
:
▄
7classification_layers/dense_last/dense/bias/Adam_1/readIdentity2classification_layers/dense_last/dense/bias/Adam_1*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
valueB
 *wЙ?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w╠+2*
_output_shapes
: *
dtype0
щ
9Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d1/conv1d/kernel(conv1D_layers/conv1d1/conv1d/kernel/Adam*conv1D_layers/conv1d1/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d1/conv1d/convolution/ExpandDims_1_grad/Reshape*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
use_nesterov( *"
_output_shapes
:
ж
7Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d1/conv1d/bias&conv1D_layers/conv1d1/conv1d/bias/Adam(conv1D_layers/conv1d1/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d1/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d1/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
щ
9Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d2/conv1d/kernel(conv1D_layers/conv1d2/conv1d/kernel/Adam*conv1D_layers/conv1d2/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d2/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d2/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d2/conv1d/bias&conv1D_layers/conv1d2/conv1d/bias/Adam(conv1D_layers/conv1d2/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d2/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d2/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d3/conv1d/kernel(conv1D_layers/conv1d3/conv1d/kernel/Adam*conv1D_layers/conv1d3/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d3/conv1d/convolution/ExpandDims_1_grad/Reshape*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d3/conv1d/kernel*
use_nesterov( *"
_output_shapes
:
ж
7Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d3/conv1d/bias&conv1D_layers/conv1d3/conv1d/bias/Adam(conv1D_layers/conv1d3/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d3/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d3/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
щ
9Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d4/conv1d/kernel(conv1D_layers/conv1d4/conv1d/kernel/Adam*conv1D_layers/conv1d4/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d4/conv1d/convolution/ExpandDims_1_grad/Reshape*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d4/conv1d/kernel*
use_nesterov( *"
_output_shapes
:
ж
7Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d4/conv1d/bias&conv1D_layers/conv1d4/conv1d/bias/Adam(conv1D_layers/conv1d4/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d4/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d4/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d5/conv1d/kernel(conv1D_layers/conv1d5/conv1d/kernel/Adam*conv1D_layers/conv1d5/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d5/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d5/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d5/conv1d/bias&conv1D_layers/conv1d5/conv1d/bias/Adam(conv1D_layers/conv1d5/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d5/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d5/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d6/conv1d/kernel(conv1D_layers/conv1d6/conv1d/kernel/Adam*conv1D_layers/conv1d6/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d6/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d6/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d6/conv1d/bias&conv1D_layers/conv1d6/conv1d/bias/Adam(conv1D_layers/conv1d6/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d6/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d6/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
щ
9Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d7/conv1d/kernel(conv1D_layers/conv1d7/conv1d/kernel/Adam*conv1D_layers/conv1d7/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d7/conv1d/convolution/ExpandDims_1_grad/Reshape*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d7/conv1d/kernel*
use_nesterov( *"
_output_shapes
:
ж
7Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d7/conv1d/bias&conv1D_layers/conv1d7/conv1d/bias/Adam(conv1D_layers/conv1d7/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d7/conv1d/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@conv1D_layers/conv1d7/conv1d/bias*
use_nesterov( *
_output_shapes
:
щ
9Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam	ApplyAdam#conv1D_layers/conv1d8/conv1d/kernel(conv1D_layers/conv1d8/conv1d/kernel/Adam*conv1D_layers/conv1d8/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/conv1D_layers/conv1d8/conv1d/convolution/ExpandDims_1_grad/Reshape*6
_class,
*(loc:@conv1D_layers/conv1d8/conv1d/kernel*"
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
ж
7Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam	ApplyAdam!conv1D_layers/conv1d8/conv1d/bias&conv1D_layers/conv1d8/conv1d/bias/Adam(conv1D_layers/conv1d8/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonNgradients/conv1D_layers/conv1d8/conv1d/BiasAdd_grad/tuple/control_dependency_1*4
_class*
(&loc:@conv1D_layers/conv1d8/conv1d/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
џ
?Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam	ApplyAdam)classification_layers/dense0/dense/kernel.classification_layers/dense0/dense/kernel/Adam0classification_layers/dense0/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes

: 
*
T0*
use_nesterov( *
use_locking( 
Ї
=Adam/update_classification_layers/dense0/dense/bias/ApplyAdam	ApplyAdam'classification_layers/dense0/dense/bias,classification_layers/dense0/dense/bias/Adam.classification_layers/dense0/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonTgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
use_nesterov( *
_output_shapes
:

▓
CAdam/update_classification_layers/dense_last/dense/kernel/ApplyAdam	ApplyAdam-classification_layers/dense_last/dense/kernel2classification_layers/dense_last/dense/kernel/Adam4classification_layers/dense_last/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonWgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
use_nesterov( *
use_locking( 
Ц
AAdam/update_classification_layers/dense_last/dense/bias/ApplyAdam	ApplyAdam+classification_layers/dense_last/dense/bias0classification_layers/dense_last/dense/bias/Adam2classification_layers/dense_last/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonXgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
use_nesterov( *
_output_shapes
:
┬

Adam/mulMulbeta1_power/read
Adam/beta1:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: 
«
Adam/AssignAssignbeta1_powerAdam/mul*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
─


Adam/mul_1Mulbeta2_power/read
Adam/beta2:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
_output_shapes
: 
▓
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*6
_class,
*(loc:@conv1D_layers/conv1d1/conv1d/kernel*
validate_shape(*
_output_shapes
: 
Т	
AdamNoOp:^Adam/update_conv1D_layers/conv1d1/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d1/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d2/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d2/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d3/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d3/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d4/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d4/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d5/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d5/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d6/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d6/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d7/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d7/conv1d/bias/ApplyAdam:^Adam/update_conv1D_layers/conv1d8/conv1d/kernel/ApplyAdam8^Adam/update_conv1D_layers/conv1d8/conv1d/bias/ApplyAdam@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
о
initNoOp+^conv1D_layers/conv1d1/conv1d/kernel/Assign)^conv1D_layers/conv1d1/conv1d/bias/Assign+^conv1D_layers/conv1d2/conv1d/kernel/Assign)^conv1D_layers/conv1d2/conv1d/bias/Assign+^conv1D_layers/conv1d3/conv1d/kernel/Assign)^conv1D_layers/conv1d3/conv1d/bias/Assign+^conv1D_layers/conv1d4/conv1d/kernel/Assign)^conv1D_layers/conv1d4/conv1d/bias/Assign+^conv1D_layers/conv1d5/conv1d/kernel/Assign)^conv1D_layers/conv1d5/conv1d/bias/Assign+^conv1D_layers/conv1d6/conv1d/kernel/Assign)^conv1D_layers/conv1d6/conv1d/bias/Assign+^conv1D_layers/conv1d7/conv1d/kernel/Assign)^conv1D_layers/conv1d7/conv1d/bias/Assign+^conv1D_layers/conv1d8/conv1d/kernel/Assign)^conv1D_layers/conv1d8/conv1d/bias/Assign1^classification_layers/dense0/dense/kernel/Assign/^classification_layers/dense0/dense/bias/Assign5^classification_layers/dense_last/dense/kernel/Assign3^classification_layers/dense_last/dense/bias/Assign^beta1_power/Assign^beta2_power/Assign0^conv1D_layers/conv1d1/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d1/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d1/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d2/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d2/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d2/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d3/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d3/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d3/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d4/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d4/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d4/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d5/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d5/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d5/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d6/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d6/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d6/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d7/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d7/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d7/conv1d/bias/Adam_1/Assign0^conv1D_layers/conv1d8/conv1d/kernel/Adam/Assign2^conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Assign.^conv1D_layers/conv1d8/conv1d/bias/Adam/Assign0^conv1D_layers/conv1d8/conv1d/bias/Adam_1/Assign6^classification_layers/dense0/dense/kernel/Adam/Assign8^classification_layers/dense0/dense/kernel/Adam_1/Assign4^classification_layers/dense0/dense/bias/Adam/Assign6^classification_layers/dense0/dense/bias/Adam_1/Assign:^classification_layers/dense_last/dense/kernel/Adam/Assign<^classification_layers/dense_last/dense/kernel/Adam_1/Assign8^classification_layers/dense_last/dense/bias/Adam/Assign:^classification_layers/dense_last/dense/bias/Adam_1/Assign""g
	summariesZ
X
Evaluation_layers/accuracy:0
Evaluation_layers/loss:0
Evaluation_layers/accuracy_1:0"О
trainable_variables┐╝

%conv1D_layers/conv1d1/conv1d/kernel:0*conv1D_layers/conv1d1/conv1d/kernel/Assign*conv1D_layers/conv1d1/conv1d/kernel/read:0
y
#conv1D_layers/conv1d1/conv1d/bias:0(conv1D_layers/conv1d1/conv1d/bias/Assign(conv1D_layers/conv1d1/conv1d/bias/read:0

%conv1D_layers/conv1d2/conv1d/kernel:0*conv1D_layers/conv1d2/conv1d/kernel/Assign*conv1D_layers/conv1d2/conv1d/kernel/read:0
y
#conv1D_layers/conv1d2/conv1d/bias:0(conv1D_layers/conv1d2/conv1d/bias/Assign(conv1D_layers/conv1d2/conv1d/bias/read:0

%conv1D_layers/conv1d3/conv1d/kernel:0*conv1D_layers/conv1d3/conv1d/kernel/Assign*conv1D_layers/conv1d3/conv1d/kernel/read:0
y
#conv1D_layers/conv1d3/conv1d/bias:0(conv1D_layers/conv1d3/conv1d/bias/Assign(conv1D_layers/conv1d3/conv1d/bias/read:0

%conv1D_layers/conv1d4/conv1d/kernel:0*conv1D_layers/conv1d4/conv1d/kernel/Assign*conv1D_layers/conv1d4/conv1d/kernel/read:0
y
#conv1D_layers/conv1d4/conv1d/bias:0(conv1D_layers/conv1d4/conv1d/bias/Assign(conv1D_layers/conv1d4/conv1d/bias/read:0

%conv1D_layers/conv1d5/conv1d/kernel:0*conv1D_layers/conv1d5/conv1d/kernel/Assign*conv1D_layers/conv1d5/conv1d/kernel/read:0
y
#conv1D_layers/conv1d5/conv1d/bias:0(conv1D_layers/conv1d5/conv1d/bias/Assign(conv1D_layers/conv1d5/conv1d/bias/read:0

%conv1D_layers/conv1d6/conv1d/kernel:0*conv1D_layers/conv1d6/conv1d/kernel/Assign*conv1D_layers/conv1d6/conv1d/kernel/read:0
y
#conv1D_layers/conv1d6/conv1d/bias:0(conv1D_layers/conv1d6/conv1d/bias/Assign(conv1D_layers/conv1d6/conv1d/bias/read:0

%conv1D_layers/conv1d7/conv1d/kernel:0*conv1D_layers/conv1d7/conv1d/kernel/Assign*conv1D_layers/conv1d7/conv1d/kernel/read:0
y
#conv1D_layers/conv1d7/conv1d/bias:0(conv1D_layers/conv1d7/conv1d/bias/Assign(conv1D_layers/conv1d7/conv1d/bias/read:0

%conv1D_layers/conv1d8/conv1d/kernel:0*conv1D_layers/conv1d8/conv1d/kernel/Assign*conv1D_layers/conv1d8/conv1d/kernel/read:0
y
#conv1D_layers/conv1d8/conv1d/bias:0(conv1D_layers/conv1d8/conv1d/bias/Assign(conv1D_layers/conv1d8/conv1d/bias/read:0
Љ
+classification_layers/dense0/dense/kernel:00classification_layers/dense0/dense/kernel/Assign0classification_layers/dense0/dense/kernel/read:0
І
)classification_layers/dense0/dense/bias:0.classification_layers/dense0/dense/bias/Assign.classification_layers/dense0/dense/bias/read:0
Ю
/classification_layers/dense_last/dense/kernel:04classification_layers/dense_last/dense/kernel/Assign4classification_layers/dense_last/dense/kernel/read:0
Ќ
-classification_layers/dense_last/dense/bias:02classification_layers/dense_last/dense/bias/Assign2classification_layers/dense_last/dense/bias/read:0"
train_op

Adam"ДD
	variablesЎDќD

%conv1D_layers/conv1d1/conv1d/kernel:0*conv1D_layers/conv1d1/conv1d/kernel/Assign*conv1D_layers/conv1d1/conv1d/kernel/read:0
y
#conv1D_layers/conv1d1/conv1d/bias:0(conv1D_layers/conv1d1/conv1d/bias/Assign(conv1D_layers/conv1d1/conv1d/bias/read:0

%conv1D_layers/conv1d2/conv1d/kernel:0*conv1D_layers/conv1d2/conv1d/kernel/Assign*conv1D_layers/conv1d2/conv1d/kernel/read:0
y
#conv1D_layers/conv1d2/conv1d/bias:0(conv1D_layers/conv1d2/conv1d/bias/Assign(conv1D_layers/conv1d2/conv1d/bias/read:0

%conv1D_layers/conv1d3/conv1d/kernel:0*conv1D_layers/conv1d3/conv1d/kernel/Assign*conv1D_layers/conv1d3/conv1d/kernel/read:0
y
#conv1D_layers/conv1d3/conv1d/bias:0(conv1D_layers/conv1d3/conv1d/bias/Assign(conv1D_layers/conv1d3/conv1d/bias/read:0

%conv1D_layers/conv1d4/conv1d/kernel:0*conv1D_layers/conv1d4/conv1d/kernel/Assign*conv1D_layers/conv1d4/conv1d/kernel/read:0
y
#conv1D_layers/conv1d4/conv1d/bias:0(conv1D_layers/conv1d4/conv1d/bias/Assign(conv1D_layers/conv1d4/conv1d/bias/read:0

%conv1D_layers/conv1d5/conv1d/kernel:0*conv1D_layers/conv1d5/conv1d/kernel/Assign*conv1D_layers/conv1d5/conv1d/kernel/read:0
y
#conv1D_layers/conv1d5/conv1d/bias:0(conv1D_layers/conv1d5/conv1d/bias/Assign(conv1D_layers/conv1d5/conv1d/bias/read:0

%conv1D_layers/conv1d6/conv1d/kernel:0*conv1D_layers/conv1d6/conv1d/kernel/Assign*conv1D_layers/conv1d6/conv1d/kernel/read:0
y
#conv1D_layers/conv1d6/conv1d/bias:0(conv1D_layers/conv1d6/conv1d/bias/Assign(conv1D_layers/conv1d6/conv1d/bias/read:0

%conv1D_layers/conv1d7/conv1d/kernel:0*conv1D_layers/conv1d7/conv1d/kernel/Assign*conv1D_layers/conv1d7/conv1d/kernel/read:0
y
#conv1D_layers/conv1d7/conv1d/bias:0(conv1D_layers/conv1d7/conv1d/bias/Assign(conv1D_layers/conv1d7/conv1d/bias/read:0

%conv1D_layers/conv1d8/conv1d/kernel:0*conv1D_layers/conv1d8/conv1d/kernel/Assign*conv1D_layers/conv1d8/conv1d/kernel/read:0
y
#conv1D_layers/conv1d8/conv1d/bias:0(conv1D_layers/conv1d8/conv1d/bias/Assign(conv1D_layers/conv1d8/conv1d/bias/read:0
Љ
+classification_layers/dense0/dense/kernel:00classification_layers/dense0/dense/kernel/Assign0classification_layers/dense0/dense/kernel/read:0
І
)classification_layers/dense0/dense/bias:0.classification_layers/dense0/dense/bias/Assign.classification_layers/dense0/dense/bias/read:0
Ю
/classification_layers/dense_last/dense/kernel:04classification_layers/dense_last/dense/kernel/Assign4classification_layers/dense_last/dense/kernel/read:0
Ќ
-classification_layers/dense_last/dense/bias:02classification_layers/dense_last/dense/bias/Assign2classification_layers/dense_last/dense/bias/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
ј
*conv1D_layers/conv1d1/conv1d/kernel/Adam:0/conv1D_layers/conv1d1/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d1/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d1/conv1d/kernel/Adam_1:01conv1D_layers/conv1d1/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d1/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d1/conv1d/bias/Adam:0-conv1D_layers/conv1d1/conv1d/bias/Adam/Assign-conv1D_layers/conv1d1/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d1/conv1d/bias/Adam_1:0/conv1D_layers/conv1d1/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d1/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d2/conv1d/kernel/Adam:0/conv1D_layers/conv1d2/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d2/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d2/conv1d/kernel/Adam_1:01conv1D_layers/conv1d2/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d2/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d2/conv1d/bias/Adam:0-conv1D_layers/conv1d2/conv1d/bias/Adam/Assign-conv1D_layers/conv1d2/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d2/conv1d/bias/Adam_1:0/conv1D_layers/conv1d2/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d2/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d3/conv1d/kernel/Adam:0/conv1D_layers/conv1d3/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d3/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d3/conv1d/kernel/Adam_1:01conv1D_layers/conv1d3/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d3/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d3/conv1d/bias/Adam:0-conv1D_layers/conv1d3/conv1d/bias/Adam/Assign-conv1D_layers/conv1d3/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d3/conv1d/bias/Adam_1:0/conv1D_layers/conv1d3/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d3/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d4/conv1d/kernel/Adam:0/conv1D_layers/conv1d4/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d4/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d4/conv1d/kernel/Adam_1:01conv1D_layers/conv1d4/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d4/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d4/conv1d/bias/Adam:0-conv1D_layers/conv1d4/conv1d/bias/Adam/Assign-conv1D_layers/conv1d4/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d4/conv1d/bias/Adam_1:0/conv1D_layers/conv1d4/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d4/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d5/conv1d/kernel/Adam:0/conv1D_layers/conv1d5/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d5/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d5/conv1d/kernel/Adam_1:01conv1D_layers/conv1d5/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d5/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d5/conv1d/bias/Adam:0-conv1D_layers/conv1d5/conv1d/bias/Adam/Assign-conv1D_layers/conv1d5/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d5/conv1d/bias/Adam_1:0/conv1D_layers/conv1d5/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d5/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d6/conv1d/kernel/Adam:0/conv1D_layers/conv1d6/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d6/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d6/conv1d/kernel/Adam_1:01conv1D_layers/conv1d6/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d6/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d6/conv1d/bias/Adam:0-conv1D_layers/conv1d6/conv1d/bias/Adam/Assign-conv1D_layers/conv1d6/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d6/conv1d/bias/Adam_1:0/conv1D_layers/conv1d6/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d6/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d7/conv1d/kernel/Adam:0/conv1D_layers/conv1d7/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d7/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d7/conv1d/kernel/Adam_1:01conv1D_layers/conv1d7/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d7/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d7/conv1d/bias/Adam:0-conv1D_layers/conv1d7/conv1d/bias/Adam/Assign-conv1D_layers/conv1d7/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d7/conv1d/bias/Adam_1:0/conv1D_layers/conv1d7/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d7/conv1d/bias/Adam_1/read:0
ј
*conv1D_layers/conv1d8/conv1d/kernel/Adam:0/conv1D_layers/conv1d8/conv1d/kernel/Adam/Assign/conv1D_layers/conv1d8/conv1d/kernel/Adam/read:0
ћ
,conv1D_layers/conv1d8/conv1d/kernel/Adam_1:01conv1D_layers/conv1d8/conv1d/kernel/Adam_1/Assign1conv1D_layers/conv1d8/conv1d/kernel/Adam_1/read:0
ѕ
(conv1D_layers/conv1d8/conv1d/bias/Adam:0-conv1D_layers/conv1d8/conv1d/bias/Adam/Assign-conv1D_layers/conv1d8/conv1d/bias/Adam/read:0
ј
*conv1D_layers/conv1d8/conv1d/bias/Adam_1:0/conv1D_layers/conv1d8/conv1d/bias/Adam_1/Assign/conv1D_layers/conv1d8/conv1d/bias/Adam_1/read:0
а
0classification_layers/dense0/dense/kernel/Adam:05classification_layers/dense0/dense/kernel/Adam/Assign5classification_layers/dense0/dense/kernel/Adam/read:0
д
2classification_layers/dense0/dense/kernel/Adam_1:07classification_layers/dense0/dense/kernel/Adam_1/Assign7classification_layers/dense0/dense/kernel/Adam_1/read:0
џ
.classification_layers/dense0/dense/bias/Adam:03classification_layers/dense0/dense/bias/Adam/Assign3classification_layers/dense0/dense/bias/Adam/read:0
а
0classification_layers/dense0/dense/bias/Adam_1:05classification_layers/dense0/dense/bias/Adam_1/Assign5classification_layers/dense0/dense/bias/Adam_1/read:0
г
4classification_layers/dense_last/dense/kernel/Adam:09classification_layers/dense_last/dense/kernel/Adam/Assign9classification_layers/dense_last/dense/kernel/Adam/read:0
▓
6classification_layers/dense_last/dense/kernel/Adam_1:0;classification_layers/dense_last/dense/kernel/Adam_1/Assign;classification_layers/dense_last/dense/kernel/Adam_1/read:0
д
2classification_layers/dense_last/dense/bias/Adam:07classification_layers/dense_last/dense/bias/Adam/Assign7classification_layers/dense_last/dense/bias/Adam/read:0
г
4classification_layers/dense_last/dense/bias/Adam_1:09classification_layers/dense_last/dense/bias/Adam_1/Assign9classification_layers/dense_last/dense/bias/Adam_1/read:0╣!Kr       %:ѓ	▀Ф <Ъ^оA*g
!
Evaluation_layers/accuracy> ?

Evaluation_layers/lossOr1?
#
Evaluation_layers/accuracy_1> ?Фбt       _gsм	J5Љ<Ъ^оA*g
!
Evaluation_layers/accuracy> ?

Evaluation_layers/lossЭq1?
#
Evaluation_layers/accuracy_1> ?ф╬ st       _gsм	 ^Э<Ъ^оA*g
!
Evaluation_layers/accuracy> ?

Evaluation_layers/loss}v1?
#
Evaluation_layers/accuracy_1> ?)Ёy├t       _gsм	хTU=Ъ^оA*g
!
Evaluation_layers/accuracy> ?

Evaluation_layers/lossК-2?
#
Evaluation_layers/accuracy_1> ?╗еM■t       _gsм	vc»=Ъ^оA*g
!
Evaluation_layers/accuracyЩ_-?

Evaluation_layers/loss~d)?
#
Evaluation_layers/accuracy_1Щ_-?пW­«t       _gsм	ЗЋ>Ъ^оA*g
!
Evaluation_layers/accuracy\ц?

Evaluation_layers/lossS╩'?
#
Evaluation_layers/accuracy_1\ц?МЗљ%t       _gsм	Э\і>Ъ^оA*g
!
Evaluation_layers/accuracy░Ц(?

Evaluation_layers/loss╦ё#?
#
Evaluation_layers/accuracy_1░Ц(?`^[t       _gsм	┐KЗ>Ъ^оA*g
!
Evaluation_layers/accuracy3#?

Evaluation_layers/lossГ#?
#
Evaluation_layers/accuracy_13#?]д4t       _gsм	1#O?Ъ^оA*g
!
Evaluation_layers/accuracy
o+?

Evaluation_layers/lossЊE?
#
Evaluation_layers/accuracy_1
o+?wСJt       _gsм	f╣Е?Ъ^оA	*g
!
Evaluation_layers/accuracySG4?

Evaluation_layers/loss;Џ?
#
Evaluation_layers/accuracy_1SG4?oЯst       _gsм	a▒7@Ъ^оA
*g
!
Evaluation_layers/accuracy;2?

Evaluation_layers/lossJ1?
#
Evaluation_layers/accuracy_1;2?­18t       _gsм	x┘Е@Ъ^оA*g
!
Evaluation_layers/accuracy┐3?

Evaluation_layers/loss6з?
#
Evaluation_layers/accuracy_1┐3?jЅe	t       _gsм	ЋАAЪ^оA*g
!
Evaluation_layers/accuracyн11?

Evaluation_layers/loss¤8?
#
Evaluation_layers/accuracy_1н11?р~MПt       _gsм	SilAЪ^оA*g
!
Evaluation_layers/accuracy;ш/?

Evaluation_layers/loss│5?
#
Evaluation_layers/accuracy_1;ш/?эhvt       _gsм	[M╚AЪ^оA*g
!
Evaluation_layers/accuracy¤X7?

Evaluation_layers/lossњЊ?
#
Evaluation_layers/accuracy_1¤X7?─Qt       _gsм	W│:BЪ^оA*g
!
Evaluation_layers/accuracyу╗=?

Evaluation_layers/loss[С
?
#
Evaluation_layers/accuracy_1у╗=?жfЂot       _gsм	Г-«BЪ^оA*g
!
Evaluation_layers/accuracyzе6?

Evaluation_layers/loss]x?
#
Evaluation_layers/accuracy_1zе6?оp.хt       _gsм	=CЪ^оA*g
!
Evaluation_layers/accuracyЛ~;?

Evaluation_layers/lossч
?
#
Evaluation_layers/accuracy_1Л~;?оwкt       _gsм	IcnCЪ^оA*g
!
Evaluation_layers/accuracyЖљ7?

Evaluation_layers/lossпL?
#
Evaluation_layers/accuracy_1Жљ7?ќ>чаt       _gsм	8Ћ¤CЪ^оA*g
!
Evaluation_layers/accuracy¤63?

Evaluation_layers/loss|i?
#
Evaluation_layers/accuracy_1¤63?ўVИЇt       _gsм	L_pDЪ^оA*g
!
Evaluation_layers/accuracycџ:?

Evaluation_layers/loss╗н?
#
Evaluation_layers/accuracy_1cџ:?в¤Џmt       _gsм	^m▀DЪ^оA*g
!
Evaluation_layers/accuracyу╗=?

Evaluation_layers/loss7H	?
#
Evaluation_layers/accuracy_1у╗=?╗ыt       _gsм	еD9EЪ^оA*g
!
Evaluation_layers/accuracyZ╩B?

Evaluation_layers/lossk)?
#
Evaluation_layers/accuracy_1Z╩B?J0Іt       _gsм	!ЎEЪ^оA*g
!
Evaluation_layers/accuracyБ@??

Evaluation_layers/loss=а?
#
Evaluation_layers/accuracy_1Б@??~r}цt       _gsм	JящEЪ^оA*g
!
Evaluation_layers/accuracyж┐=?

Evaluation_layers/lossvа	?
#
Evaluation_layers/accuracy_1ж┐=?lk[t       _gsм	й@mFЪ^оA*g
!
Evaluation_layers/accuracy@c<?

Evaluation_layers/loss(p?
#
Evaluation_layers/accuracy_1@c<?BA"t       _gsм	\дПFЪ^оA*g
!
Evaluation_layers/accuracydяB?

Evaluation_layers/lossё:?
#
Evaluation_layers/accuracy_1dяB?╠јt       _gsм	">7GЪ^оA*g
!
Evaluation_layers/accuracyv┬:?

Evaluation_layers/lossPX?
#
Evaluation_layers/accuracy_1v┬:?Нцгt       _gsм	'вЉGЪ^оA*g
!
Evaluation_layers/accuracy02:?

Evaluation_layers/lossрљ?
#
Evaluation_layers/accuracy_102:?bюЪВt       _gsм	\шыGЪ^оA*g
!
Evaluation_layers/accuracy%+<?

Evaluation_layers/lossmК	?
#
Evaluation_layers/accuracy_1%+<?ўІat       _gsм	яЂЄHЪ^оA*g
!
Evaluation_layers/accuracyу╗=?

Evaluation_layers/loss2[?
#
Evaluation_layers/accuracy_1у╗=?ЩZщюt       _gsм	=љЫHЪ^оA*g
!
Evaluation_layers/accuracyNB?

Evaluation_layers/lossЗм ?
#
Evaluation_layers/accuracy_1NB?д╦К╬t       _gsм	њNIЪ^оA *g
!
Evaluation_layers/accuracy║N;?

Evaluation_layers/lossGб?
#
Evaluation_layers/accuracy_1║N;?nЇЈt       _gsм	CЈЕIЪ^оA!*g
!
Evaluation_layers/accuracy@c<?

Evaluation_layers/lossюЖ?
#
Evaluation_layers/accuracy_1@c<?аP┼ut       _gsм	▒JЪ^оA"*g
!
Evaluation_layers/accuracy>?

Evaluation_layers/loss▒─?
#
Evaluation_layers/accuracy_1>?ъПжрt       _gsм	јБѓJЪ^оA#*g
!
Evaluation_layers/accuracyv┬:?

Evaluation_layers/loss@Ѕ?
#
Evaluation_layers/accuracy_1v┬:?GT╗ћ