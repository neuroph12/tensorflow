       ЃK"	  ќЃ^жAbrain.Event:2f>n`Љ4     \­ы	h!ќЃ^жA"щ
|
Input/PlaceholderPlaceholder*+
_output_shapes
:џџџџџџџџџ * 
shape:џџџџџџџџџ *
dtype0
u
Target/PlaceholderPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
g
"controll_normalization/PlaceholderPlaceholder*
shape:*
dtype0
*
_output_shapes
:
^
Flatten/ShapeShapeInput/Placeholder*
T0*
out_type0*
_output_shapes
:
]
Flatten/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
\
Flatten/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:

Flatten/SliceSliceFlatten/ShapeFlatten/Slice/beginFlatten/Slice/size*
T0*
Index0*
_output_shapes
:
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

Flatten/Slice_1SliceFlatten/ShapeFlatten/Slice_1/beginFlatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
W
Flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
Flatten/ProdProdFlatten/Slice_1Flatten/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
w
Flatten/ExpandDims
ExpandDimsFlatten/ProdFlatten/ExpandDims/dim*
_output_shapes
:*
T0*

Tdim0
U
Flatten/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0

Flatten/concatConcatV2Flatten/SliceFlatten/ExpandDimsFlatten/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
~
Flatten/ReshapeReshapeInput/PlaceholderFlatten/concat*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
f
!classification_layers/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0
л
Lclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB"   
   
Ю
Kclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *    
а
Mclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *  ?
Х
Vclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shape*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*

seed *
T0*
seed2 
р
Jclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulMulVclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalMclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddev*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0
Ю
Fclassification_layers/dense0/dense/kernel/Initializer/truncated_normalAddJclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulKclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mean*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	

н
)classification_layers/dense0/dense/kernel
VariableV2*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	
*
shape:	
*
dtype0*
shared_name *
	container 
О
0classification_layers/dense0/dense/kernel/AssignAssign)classification_layers/dense0/dense/kernelFclassification_layers/dense0/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
:	

Э
.classification_layers/dense0/dense/kernel/readIdentity)classification_layers/dense0/dense/kernel*
T0*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel
Т
9classification_layers/dense0/dense/bias/Initializer/zerosConst*
_output_shapes
:
*
dtype0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    
Я
'classification_layers/dense0/dense/bias
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
shared_name 
І
.classification_layers/dense0/dense/bias/AssignAssign'classification_layers/dense0/dense/bias9classification_layers/dense0/dense/bias/Initializer/zeros*
_output_shapes
:
*
validate_shape(*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
T0*
use_locking(
Т
,classification_layers/dense0/dense/bias/readIdentity'classification_layers/dense0/dense/bias*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
T0
Ь
)classification_layers/dense0/dense/MatMulMatMulFlatten/Reshape.classification_layers/dense0/dense/kernel/read*
transpose_b( *'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
T0
з
*classification_layers/dense0/dense/BiasAddBiasAdd)classification_layers/dense0/dense/MatMul,classification_layers/dense0/dense/bias/read*'
_output_shapes
:џџџџџџџџџ
*
data_formatNHWC*
T0
о
Gclassification_layers/dense0/batch_normalization/beta/Initializer/zerosConst*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    *
_output_shapes
:
*
dtype0
ы
5classification_layers/dense0/batch_normalization/beta
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
dtype0*
	container 
о
<classification_layers/dense0/batch_normalization/beta/AssignAssign5classification_layers/dense0/batch_normalization/betaGclassification_layers/dense0/batch_normalization/beta/Initializer/zeros*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
ь
:classification_layers/dense0/batch_normalization/beta/readIdentity5classification_layers/dense0/batch_normalization/beta*
T0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:

п
Gclassification_layers/dense0/batch_normalization/gamma/Initializer/onesConst*
_output_shapes
:
*
dtype0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*  ?
э
6classification_layers/dense0/batch_normalization/gamma
VariableV2*
	container *
dtype0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
shared_name *
_output_shapes
:
*
shape:

с
=classification_layers/dense0/batch_normalization/gamma/AssignAssign6classification_layers/dense0/batch_normalization/gammaGclassification_layers/dense0/batch_normalization/gamma/Initializer/ones*
_output_shapes
:
*
validate_shape(*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
T0*
use_locking(
я
;classification_layers/dense0/batch_normalization/gamma/readIdentity6classification_layers/dense0/batch_normalization/gamma*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:
*
T0
ь
Nclassification_layers/dense0/batch_normalization/moving_mean/Initializer/zerosConst*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
valueB
*    *
dtype0*
_output_shapes
:

љ
<classification_layers/dense0/batch_normalization/moving_mean
VariableV2*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
њ
Cclassification_layers/dense0/batch_normalization/moving_mean/AssignAssign<classification_layers/dense0/batch_normalization/moving_meanNclassification_layers/dense0/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:


Aclassification_layers/dense0/batch_normalization/moving_mean/readIdentity<classification_layers/dense0/batch_normalization/moving_mean*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:

ѓ
Qclassification_layers/dense0/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes
:
*
dtype0*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
valueB
*  ?

@classification_layers/dense0/batch_normalization/moving_variance
VariableV2*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 

Gclassification_layers/dense0/batch_normalization/moving_variance/AssignAssign@classification_layers/dense0/batch_normalization/moving_varianceQclassification_layers/dense0/batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:
*
validate_shape(*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
T0*
use_locking(

Eclassification_layers/dense0/batch_normalization/moving_variance/readIdentity@classification_layers/dense0/batch_normalization/moving_variance*
T0*
_output_shapes
:
*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance

Oclassification_layers/dense0/batch_normalization/moments/Mean/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0

=classification_layers/dense0/batch_normalization/moments/MeanMean*classification_layers/dense0/dense/BiasAddOclassification_layers/dense0/batch_normalization/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes

:

Н
Eclassification_layers/dense0/batch_normalization/moments/StopGradientStopGradient=classification_layers/dense0/batch_normalization/moments/Mean*
T0*
_output_shapes

:

ш
<classification_layers/dense0/batch_normalization/moments/SubSub*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ
*
T0
Ё
Wclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Њ
Eclassification_layers/dense0/batch_normalization/moments/shifted_meanMean<classification_layers/dense0/batch_normalization/moments/SubWclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes

:


Jclassification_layers/dense0/batch_normalization/moments/SquaredDifferenceSquaredDifference*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ


Qclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
Ќ
?classification_layers/dense0/batch_normalization/moments/Mean_1MeanJclassification_layers/dense0/batch_normalization/moments/SquaredDifferenceQclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes

:

Й
?classification_layers/dense0/batch_normalization/moments/SquareSquareEclassification_layers/dense0/batch_normalization/moments/shifted_mean*
_output_shapes

:
*
T0
ѓ
Aclassification_layers/dense0/batch_normalization/moments/varianceSub?classification_layers/dense0/batch_normalization/moments/Mean_1?classification_layers/dense0/batch_normalization/moments/Square*
_output_shapes

:
*
T0
ћ
=classification_layers/dense0/batch_normalization/moments/meanAddEclassification_layers/dense0/batch_normalization/moments/shifted_meanEclassification_layers/dense0/batch_normalization/moments/StopGradient*
T0*
_output_shapes

:

Ц
@classification_layers/dense0/batch_normalization/moments/SqueezeSqueeze=classification_layers/dense0/batch_normalization/moments/mean*
T0*
_output_shapes
:
*
squeeze_dims
 
Ь
Bclassification_layers/dense0/batch_normalization/moments/Squeeze_1SqueezeAclassification_layers/dense0/batch_normalization/moments/variance*
_output_shapes
:
*
T0*
squeeze_dims
 

?classification_layers/dense0/batch_normalization/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

;classification_layers/dense0/batch_normalization/ExpandDims
ExpandDims@classification_layers/dense0/batch_normalization/moments/Squeeze?classification_layers/dense0/batch_normalization/ExpandDims/dim*

Tdim0*
_output_shapes

:
*
T0

Aclassification_layers/dense0/batch_normalization/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 

=classification_layers/dense0/batch_normalization/ExpandDims_1
ExpandDimsAclassification_layers/dense0/batch_normalization/moving_mean/readAclassification_layers/dense0/batch_normalization/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:


>classification_layers/dense0/batch_normalization/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
к
8classification_layers/dense0/batch_normalization/ReshapeReshape"controll_normalization/Placeholder>classification_layers/dense0/batch_normalization/Reshape/shape*
_output_shapes
:*
Tshape0*
T0

 
7classification_layers/dense0/batch_normalization/SelectSelect8classification_layers/dense0/batch_normalization/Reshape;classification_layers/dense0/batch_normalization/ExpandDims=classification_layers/dense0/batch_normalization/ExpandDims_1*
_output_shapes

:
*
T0
И
8classification_layers/dense0/batch_normalization/SqueezeSqueeze7classification_layers/dense0/batch_normalization/Select*
T0*
_output_shapes
:
*
squeeze_dims
 

Aclassification_layers/dense0/batch_normalization/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 

=classification_layers/dense0/batch_normalization/ExpandDims_2
ExpandDimsBclassification_layers/dense0/batch_normalization/moments/Squeeze_1Aclassification_layers/dense0/batch_normalization/ExpandDims_2/dim*
_output_shapes

:
*
T0*

Tdim0

Aclassification_layers/dense0/batch_normalization/ExpandDims_3/dimConst*
value	B : *
_output_shapes
: *
dtype0

=classification_layers/dense0/batch_normalization/ExpandDims_3
ExpandDimsEclassification_layers/dense0/batch_normalization/moving_variance/readAclassification_layers/dense0/batch_normalization/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes

:


@classification_layers/dense0/batch_normalization/Reshape_1/shapeConst*
valueB:*
_output_shapes
:*
dtype0
о
:classification_layers/dense0/batch_normalization/Reshape_1Reshape"controll_normalization/Placeholder@classification_layers/dense0/batch_normalization/Reshape_1/shape*
_output_shapes
:*
Tshape0*
T0

І
9classification_layers/dense0/batch_normalization/Select_1Select:classification_layers/dense0/batch_normalization/Reshape_1=classification_layers/dense0/batch_normalization/ExpandDims_2=classification_layers/dense0/batch_normalization/ExpandDims_3*
T0*
_output_shapes

:

М
:classification_layers/dense0/batch_normalization/Squeeze_1Squeeze9classification_layers/dense0/batch_normalization/Select_1*
squeeze_dims
 *
T0*
_output_shapes
:


Cclassification_layers/dense0/batch_normalization/ExpandDims_4/inputConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?

Aclassification_layers/dense0/batch_normalization/ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
value	B : 

=classification_layers/dense0/batch_normalization/ExpandDims_4
ExpandDimsCclassification_layers/dense0/batch_normalization/ExpandDims_4/inputAclassification_layers/dense0/batch_normalization/ExpandDims_4/dim*

Tdim0*
_output_shapes
:*
T0

Cclassification_layers/dense0/batch_normalization/ExpandDims_5/inputConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Aclassification_layers/dense0/batch_normalization/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B : 

=classification_layers/dense0/batch_normalization/ExpandDims_5
ExpandDimsCclassification_layers/dense0/batch_normalization/ExpandDims_5/inputAclassification_layers/dense0/batch_normalization/ExpandDims_5/dim*

Tdim0*
_output_shapes
:*
T0

@classification_layers/dense0/batch_normalization/Reshape_2/shapeConst*
valueB:*
dtype0*
_output_shapes
:
о
:classification_layers/dense0/batch_normalization/Reshape_2Reshape"controll_normalization/Placeholder@classification_layers/dense0/batch_normalization/Reshape_2/shape*
T0
*
Tshape0*
_output_shapes
:
Ђ
9classification_layers/dense0/batch_normalization/Select_2Select:classification_layers/dense0/batch_normalization/Reshape_2=classification_layers/dense0/batch_normalization/ExpandDims_4=classification_layers/dense0/batch_normalization/ExpandDims_5*
T0*
_output_shapes
:
И
:classification_layers/dense0/batch_normalization/Squeeze_2Squeeze9classification_layers/dense0/batch_normalization/Select_2*
_output_shapes
: *
T0*
squeeze_dims
 
м
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean
С
Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/subSubFclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub/x:classification_layers/dense0/batch_normalization/Squeeze_2*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
: 
Р
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub_1SubAclassification_layers/dense0/batch_normalization/moving_mean/read8classification_layers/dense0/batch_normalization/Squeeze*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:

Я
Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/mulMulFclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub_1Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:

к
@classification_layers/dense0/batch_normalization/AssignMovingAvg	AssignSub<classification_layers/dense0/batch_normalization/moving_meanDclassification_layers/dense0/batch_normalization/AssignMovingAvg/mul*
use_locking( *
T0*
_output_shapes
:
*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean
т
Hclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub/xConst*
valueB
 *  ?*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
dtype0*
_output_shapes
: 
Щ
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/subSubHclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub/x:classification_layers/dense0/batch_normalization/Squeeze_2*
_output_shapes
: *S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
T0
Ь
Hclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub_1SubEclassification_layers/dense0/batch_normalization/moving_variance/read:classification_layers/dense0/batch_normalization/Squeeze_1*
T0*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:

й
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/mulMulHclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub_1Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub*
T0*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:

ц
Bclassification_layers/dense0/batch_normalization/AssignMovingAvg_1	AssignSub@classification_layers/dense0/batch_normalization/moving_varianceFclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/mul*
use_locking( *
T0*
_output_shapes
:
*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance

@classification_layers/dense0/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ш
>classification_layers/dense0/batch_normalization/batchnorm/addAdd:classification_layers/dense0/batch_normalization/Squeeze_1@classification_layers/dense0/batch_normalization/batchnorm/add/y*
_output_shapes
:
*
T0
Ў
@classification_layers/dense0/batch_normalization/batchnorm/RsqrtRsqrt>classification_layers/dense0/batch_normalization/batchnorm/add*
T0*
_output_shapes
:

щ
>classification_layers/dense0/batch_normalization/batchnorm/mulMul@classification_layers/dense0/batch_normalization/batchnorm/Rsqrt;classification_layers/dense0/batch_normalization/gamma/read*
_output_shapes
:
*
T0
х
@classification_layers/dense0/batch_normalization/batchnorm/mul_1Mul*classification_layers/dense0/dense/BiasAdd>classification_layers/dense0/batch_normalization/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ
*
T0
ц
@classification_layers/dense0/batch_normalization/batchnorm/mul_2Mul8classification_layers/dense0/batch_normalization/Squeeze>classification_layers/dense0/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:

ш
>classification_layers/dense0/batch_normalization/batchnorm/subSub:classification_layers/dense0/batch_normalization/beta/read@classification_layers/dense0/batch_normalization/batchnorm/mul_2*
_output_shapes
:
*
T0
ћ
@classification_layers/dense0/batch_normalization/batchnorm/add_1Add@classification_layers/dense0/batch_normalization/batchnorm/mul_1>classification_layers/dense0/batch_normalization/batchnorm/sub*'
_output_shapes
:џџџџџџџџџ
*
T0

!classification_layers/dense0/ReluRelu@classification_layers/dense0/batch_normalization/batchnorm/add_1*'
_output_shapes
:џџџџџџџџџ
*
T0

*classification_layers/dense0/dropout/ShapeShape!classification_layers/dense0/Relu*
out_type0*
_output_shapes
:*
T0
|
7classification_layers/dense0/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
|
7classification_layers/dense0/dropout/random_uniform/maxConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
ж
Aclassification_layers/dense0/dropout/random_uniform/RandomUniformRandomUniform*classification_layers/dense0/dropout/Shape*'
_output_shapes
:џџџџџџџџџ
*
seed2 *
dtype0*
T0*

seed 
б
7classification_layers/dense0/dropout/random_uniform/subSub7classification_layers/dense0/dropout/random_uniform/max7classification_layers/dense0/dropout/random_uniform/min*
_output_shapes
: *
T0
ь
7classification_layers/dense0/dropout/random_uniform/mulMulAclassification_layers/dense0/dropout/random_uniform/RandomUniform7classification_layers/dense0/dropout/random_uniform/sub*'
_output_shapes
:џџџџџџџџџ
*
T0
о
3classification_layers/dense0/dropout/random_uniformAdd7classification_layers/dense0/dropout/random_uniform/mul7classification_layers/dense0/dropout/random_uniform/min*'
_output_shapes
:џџџџџџџџџ
*
T0
Њ
(classification_layers/dense0/dropout/addAdd!classification_layers/Placeholder3classification_layers/dense0/dropout/random_uniform*
T0*
_output_shapes
:

*classification_layers/dense0/dropout/FloorFloor(classification_layers/dense0/dropout/add*
T0*
_output_shapes
:

(classification_layers/dense0/dropout/divRealDiv!classification_layers/dense0/Relu!classification_layers/Placeholder*
T0*
_output_shapes
:
З
(classification_layers/dense0/dropout/mulMul(classification_layers/dense0/dropout/div*classification_layers/dense0/dropout/Floor*'
_output_shapes
:џџџџџџџџџ
*
T0
у
Pclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB"
      
ж
Oclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/meanConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
и
Qclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *  ?
а
Zclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalPclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*

seed *
_output_shapes

:
*
T0
я
Nclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulMulZclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalQclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddev*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
н
Jclassification_layers/dense_last/dense/kernel/Initializer/truncated_normalAddNclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulOclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mean*
T0*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
у
-classification_layers/dense_last/dense/kernel
VariableV2*
_output_shapes

:
*
dtype0*
shape
:
*
	container *@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
shared_name 
Э
4classification_layers/dense_last/dense/kernel/AssignAssign-classification_layers/dense_last/dense/kernelJclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal*
_output_shapes

:
*
validate_shape(*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
T0*
use_locking(
и
2classification_layers/dense_last/dense/kernel/readIdentity-classification_layers/dense_last/dense/kernel*
T0*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
Ъ
=classification_layers/dense_last/dense/bias/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
з
+classification_layers/dense_last/dense/bias
VariableV2*
	container *
dtype0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
shared_name *
_output_shapes
:*
shape:
Ж
2classification_layers/dense_last/dense/bias/AssignAssign+classification_layers/dense_last/dense/bias=classification_layers/dense_last/dense/bias/Initializer/zeros*
_output_shapes
:*
validate_shape(*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
T0*
use_locking(
Ю
0classification_layers/dense_last/dense/bias/readIdentity+classification_layers/dense_last/dense/bias*
T0*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias
э
-classification_layers/dense_last/dense/MatMulMatMul(classification_layers/dense0/dropout/mul2classification_layers/dense_last/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
у
.classification_layers/dense_last/dense/BiasAddBiasAdd-classification_layers/dense_last/dense/MatMul0classification_layers/dense_last/dense/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ

classification_layers/SoftmaxSoftmax.classification_layers/dense_last/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
n
)Evaluation_layers/clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ў
'Evaluation_layers/clip_by_value/MinimumMinimumclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!Evaluation_layers/clip_by_value/yConst*
valueB
 *џцл.*
dtype0*
_output_shapes
: 
Ј
Evaluation_layers/clip_by_valueMaximum'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*'
_output_shapes
:џџџџџџџџџ*
T0
o
Evaluation_layers/LogLogEvaluation_layers/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
y
Evaluation_layers/mulMulTarget/PlaceholderEvaluation_layers/Log*
T0*'
_output_shapes
:џџџџџџџџџ
q
'Evaluation_layers/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ї
Evaluation_layers/SumSumEvaluation_layers/mul'Evaluation_layers/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
a
Evaluation_layers/NegNegEvaluation_layers/Sum*#
_output_shapes
:џџџџџџџџџ*
T0
a
Evaluation_layers/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Evaluation_layers/MeanMeanEvaluation_layers/NegEvaluation_layers/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
d
"Evaluation_layers/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

Evaluation_layers/ArgMaxArgMaxclassification_layers/Softmax"Evaluation_layers/ArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
f
$Evaluation_layers/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :

Evaluation_layers/ArgMax_1ArgMaxTarget/Placeholder$Evaluation_layers/ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

Evaluation_layers/EqualEqualEvaluation_layers/ArgMaxEvaluation_layers/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
|
Evaluation_layers/accracy/CastCastEvaluation_layers/Equal*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
i
Evaluation_layers/accracy/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ѕ
Evaluation_layers/accracy/MeanMeanEvaluation_layers/accracy/CastEvaluation_layers/accracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
Evaluation_layers/accuracy/tagsConst*
_output_shapes
: *
dtype0*+
value"B  BEvaluation_layers/accuracy

Evaluation_layers/accuracyScalarSummaryEvaluation_layers/accuracy/tagsEvaluation_layers/accracy/Mean*
T0*
_output_shapes
: 
r
Evaluation_layers/loss/tagsConst*'
valueB BEvaluation_layers/loss*
_output_shapes
: *
dtype0
}
Evaluation_layers/lossScalarSummaryEvaluation_layers/loss/tagsEvaluation_layers/Mean*
T0*
_output_shapes
: 
~
!Evaluation_layers/accuracy_1/tagsConst*
_output_shapes
: *
dtype0*-
value$B" BEvaluation_layers/accuracy_1

Evaluation_layers/accuracy_1ScalarSummary!Evaluation_layers/accuracy_1/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
}
3gradients/Evaluation_layers/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
А
-gradients/Evaluation_layers/Mean_grad/ReshapeReshapegradients/Fill3gradients/Evaluation_layers/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0

+gradients/Evaluation_layers/Mean_grad/ShapeShapeEvaluation_layers/Neg*
_output_shapes
:*
out_type0*
T0
Ю
*gradients/Evaluation_layers/Mean_grad/TileTile-gradients/Evaluation_layers/Mean_grad/Reshape+gradients/Evaluation_layers/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ

-gradients/Evaluation_layers/Mean_grad/Shape_1ShapeEvaluation_layers/Neg*
T0*
out_type0*
_output_shapes
:
p
-gradients/Evaluation_layers/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
u
+gradients/Evaluation_layers/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ь
*gradients/Evaluation_layers/Mean_grad/ProdProd-gradients/Evaluation_layers/Mean_grad/Shape_1+gradients/Evaluation_layers/Mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
w
-gradients/Evaluation_layers/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
а
,gradients/Evaluation_layers/Mean_grad/Prod_1Prod-gradients/Evaluation_layers/Mean_grad/Shape_2-gradients/Evaluation_layers/Mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
q
/gradients/Evaluation_layers/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
И
-gradients/Evaluation_layers/Mean_grad/MaximumMaximum,gradients/Evaluation_layers/Mean_grad/Prod_1/gradients/Evaluation_layers/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Ж
.gradients/Evaluation_layers/Mean_grad/floordivFloorDiv*gradients/Evaluation_layers/Mean_grad/Prod-gradients/Evaluation_layers/Mean_grad/Maximum*
T0*
_output_shapes
: 

*gradients/Evaluation_layers/Mean_grad/CastCast.gradients/Evaluation_layers/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
О
-gradients/Evaluation_layers/Mean_grad/truedivRealDiv*gradients/Evaluation_layers/Mean_grad/Tile*gradients/Evaluation_layers/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

(gradients/Evaluation_layers/Neg_grad/NegNeg-gradients/Evaluation_layers/Mean_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ

*gradients/Evaluation_layers/Sum_grad/ShapeShapeEvaluation_layers/mul*
T0*
_output_shapes
:*
out_type0
k
)gradients/Evaluation_layers/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
Ј
(gradients/Evaluation_layers/Sum_grad/addAdd'Evaluation_layers/Sum/reduction_indices)gradients/Evaluation_layers/Sum_grad/Size*
_output_shapes
:*
T0
Ў
(gradients/Evaluation_layers/Sum_grad/modFloorMod(gradients/Evaluation_layers/Sum_grad/add)gradients/Evaluation_layers/Sum_grad/Size*
_output_shapes
:*
T0
v
,gradients/Evaluation_layers/Sum_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
r
0gradients/Evaluation_layers/Sum_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
r
0gradients/Evaluation_layers/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ъ
*gradients/Evaluation_layers/Sum_grad/rangeRange0gradients/Evaluation_layers/Sum_grad/range/start)gradients/Evaluation_layers/Sum_grad/Size0gradients/Evaluation_layers/Sum_grad/range/delta*

Tidx0*
_output_shapes
:
q
/gradients/Evaluation_layers/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
Е
)gradients/Evaluation_layers/Sum_grad/FillFill,gradients/Evaluation_layers/Sum_grad/Shape_1/gradients/Evaluation_layers/Sum_grad/Fill/value*
_output_shapes
:*
T0
Ї
2gradients/Evaluation_layers/Sum_grad/DynamicStitchDynamicStitch*gradients/Evaluation_layers/Sum_grad/range(gradients/Evaluation_layers/Sum_grad/mod*gradients/Evaluation_layers/Sum_grad/Shape)gradients/Evaluation_layers/Sum_grad/Fill*
N*
T0*#
_output_shapes
:џџџџџџџџџ
p
.gradients/Evaluation_layers/Sum_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
Щ
,gradients/Evaluation_layers/Sum_grad/MaximumMaximum2gradients/Evaluation_layers/Sum_grad/DynamicStitch.gradients/Evaluation_layers/Sum_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ
И
-gradients/Evaluation_layers/Sum_grad/floordivFloorDiv*gradients/Evaluation_layers/Sum_grad/Shape,gradients/Evaluation_layers/Sum_grad/Maximum*
_output_shapes
:*
T0
Ц
,gradients/Evaluation_layers/Sum_grad/ReshapeReshape(gradients/Evaluation_layers/Neg_grad/Neg2gradients/Evaluation_layers/Sum_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
в
)gradients/Evaluation_layers/Sum_grad/TileTile,gradients/Evaluation_layers/Sum_grad/Reshape-gradients/Evaluation_layers/Sum_grad/floordiv*'
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0
|
*gradients/Evaluation_layers/mul_grad/ShapeShapeTarget/Placeholder*
out_type0*
_output_shapes
:*
T0

,gradients/Evaluation_layers/mul_grad/Shape_1ShapeEvaluation_layers/Log*
T0*
_output_shapes
:*
out_type0
ъ
:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Evaluation_layers/mul_grad/Shape,gradients/Evaluation_layers/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѓ
(gradients/Evaluation_layers/mul_grad/mulMul)gradients/Evaluation_layers/Sum_grad/TileEvaluation_layers/Log*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients/Evaluation_layers/mul_grad/SumSum(gradients/Evaluation_layers/mul_grad/mul:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
,gradients/Evaluation_layers/mul_grad/ReshapeReshape(gradients/Evaluation_layers/mul_grad/Sum*gradients/Evaluation_layers/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
Ђ
*gradients/Evaluation_layers/mul_grad/mul_1MulTarget/Placeholder)gradients/Evaluation_layers/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
л
*gradients/Evaluation_layers/mul_grad/Sum_1Sum*gradients/Evaluation_layers/mul_grad/mul_1<gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
г
.gradients/Evaluation_layers/mul_grad/Reshape_1Reshape*gradients/Evaluation_layers/mul_grad/Sum_1,gradients/Evaluation_layers/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

5gradients/Evaluation_layers/mul_grad/tuple/group_depsNoOp-^gradients/Evaluation_layers/mul_grad/Reshape/^gradients/Evaluation_layers/mul_grad/Reshape_1
Ђ
=gradients/Evaluation_layers/mul_grad/tuple/control_dependencyIdentity,gradients/Evaluation_layers/mul_grad/Reshape6^gradients/Evaluation_layers/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*?
_class5
31loc:@gradients/Evaluation_layers/mul_grad/Reshape*
T0
Ј
?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1Identity.gradients/Evaluation_layers/mul_grad/Reshape_16^gradients/Evaluation_layers/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*A
_class7
53loc:@gradients/Evaluation_layers/mul_grad/Reshape_1
в
/gradients/Evaluation_layers/Log_grad/Reciprocal
ReciprocalEvaluation_layers/clip_by_value@^gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
г
(gradients/Evaluation_layers/Log_grad/mulMul?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1/gradients/Evaluation_layers/Log_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0

4gradients/Evaluation_layers/clip_by_value_grad/ShapeShape'Evaluation_layers/clip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0
y
6gradients/Evaluation_layers/clip_by_value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 

6gradients/Evaluation_layers/clip_by_value_grad/Shape_2Shape(gradients/Evaluation_layers/Log_grad/mul*
T0*
_output_shapes
:*
out_type0

:gradients/Evaluation_layers/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
т
4gradients/Evaluation_layers/clip_by_value_grad/zerosFill6gradients/Evaluation_layers/clip_by_value_grad/Shape_2:gradients/Evaluation_layers/clip_by_value_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqualGreaterEqual'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*'
_output_shapes
:џџџџџџџџџ*
T0

Dgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/Evaluation_layers/clip_by_value_grad/Shape6gradients/Evaluation_layers/clip_by_value_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

5gradients/Evaluation_layers/clip_by_value_grad/SelectSelect;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
­
9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot
LogicalNot;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

7gradients/Evaluation_layers/clip_by_value_grad/Select_1Select9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
і
2gradients/Evaluation_layers/clip_by_value_grad/SumSum5gradients/Evaluation_layers/clip_by_value_grad/SelectDgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ы
6gradients/Evaluation_layers/clip_by_value_grad/ReshapeReshape2gradients/Evaluation_layers/clip_by_value_grad/Sum4gradients/Evaluation_layers/clip_by_value_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
ќ
4gradients/Evaluation_layers/clip_by_value_grad/Sum_1Sum7gradients/Evaluation_layers/clip_by_value_grad/Select_1Fgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
р
8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1Reshape4gradients/Evaluation_layers/clip_by_value_grad/Sum_16gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Л
?gradients/Evaluation_layers/clip_by_value_grad/tuple/group_depsNoOp7^gradients/Evaluation_layers/clip_by_value_grad/Reshape9^gradients/Evaluation_layers/clip_by_value_grad/Reshape_1
Ъ
Ggradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependencyIdentity6gradients/Evaluation_layers/clip_by_value_grad/Reshape@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*I
_class?
=;loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape*
T0
П
Igradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency_1Identity8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*K
_classA
?=loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape_1*
_output_shapes
: *
T0

<gradients/Evaluation_layers/clip_by_value/Minimum_grad/ShapeShapeclassification_layers/Softmax*
T0*
_output_shapes
:*
out_type0

>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
Х
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2ShapeGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency*
_output_shapes
:*
out_type0*
T0

Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
њ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zerosFill>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/Const*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual	LessEqualclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*'
_output_shapes
:џџџџџџџџџ*
T0
 
Lgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Т
=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectSelect@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqualGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
К
Agradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNot
LogicalNot@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual*'
_output_shapes
:џџџџџџџџџ
Х
?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1SelectAgradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNotGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0

:gradients/Evaluation_layers/clip_by_value/Minimum_grad/SumSum=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectLgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeReshape:gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1Sum?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1Ngradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ј
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1Reshape<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
г
Ggradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_depsNoOp?^gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeA^gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1
ъ
Ogradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeH^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*Q
_classG
ECloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape*
T0
п
Qgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1H^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
й
0gradients/classification_layers/Softmax_grad/mulMulOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyclassification_layers/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0

Bgradients/classification_layers/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
ј
0gradients/classification_layers/Softmax_grad/SumSum0gradients/classification_layers/Softmax_grad/mulBgradients/classification_layers/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0

:gradients/classification_layers/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
э
4gradients/classification_layers/Softmax_grad/ReshapeReshape0gradients/classification_layers/Softmax_grad/Sum:gradients/classification_layers/Softmax_grad/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
№
0gradients/classification_layers/Softmax_grad/subSubOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency4gradients/classification_layers/Softmax_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
М
2gradients/classification_layers/Softmax_grad/mul_1Mul0gradients/classification_layers/Softmax_grad/subclassification_layers/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ш
Igradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/classification_layers/Softmax_grad/mul_1*
data_formatNHWC*
T0*
_output_shapes
:
з
Ngradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/classification_layers/Softmax_grad/mul_1J^gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad
р
Vgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/classification_layers/Softmax_grad/mul_1O^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*E
_class;
97loc:@gradients/classification_layers/Softmax_grad/mul_1*
T0

Xgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradO^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
Б
Cgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulMatMulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency2classification_layers/dense_last/dense/kernel/read*
transpose_b(*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
T0
 
Egradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1MatMul(classification_layers/dense0/dropout/mulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
у
Mgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_depsNoOpD^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMulF^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1

Ugradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependencyIdentityCgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulN^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*V
_classL
JHloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul
§
Wgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1N^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:
*X
_classN
LJloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1*
T0
Ў
=gradients/classification_layers/dense0/dropout/mul_grad/ShapeShape(classification_layers/dense0/dropout/div*#
_output_shapes
:џџџџџџџџџ*
out_type0*
T0
В
?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1Shape*classification_layers/dense0/dropout/Floor*#
_output_shapes
:џџџџџџџџџ*
out_type0*
T0
Ѓ
Mgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/mul_grad/Shape?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ш
;gradients/classification_layers/dense0/dropout/mul_grad/mulMulUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*classification_layers/dense0/dropout/Floor*
_output_shapes
:*
T0

;gradients/classification_layers/dense0/dropout/mul_grad/SumSum;gradients/classification_layers/dense0/dropout/mul_grad/mulMgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ї
?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/mul_grad/Sum=gradients/classification_layers/dense0/dropout/mul_grad/Shape*
T0*
_output_shapes
:*
Tshape0
ш
=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Mul(classification_layers/dense0/dropout/divUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:

=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1Sum=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Ogradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
§
Agradients/classification_layers/dense0/dropout/mul_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
ж
Hgradients/classification_layers/dense0/dropout/mul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/mul_grad/ReshapeB^gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1
п
Pgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeI^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape*
T0
х
Rgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/mul_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0

=gradients/classification_layers/dense0/dropout/div_grad/ShapeShape!classification_layers/dense0/Relu*
T0*
out_type0*
_output_shapes
:
Љ
?gradients/classification_layers/dense0/dropout/div_grad/Shape_1Shape!classification_layers/Placeholder*
T0*#
_output_shapes
:џџџџџџџџџ*
out_type0
Ѓ
Mgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/div_grad/Shape?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
т
?gradients/classification_layers/dense0/dropout/div_grad/RealDivRealDivPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency!classification_layers/Placeholder*
T0*
_output_shapes
:

;gradients/classification_layers/dense0/dropout/div_grad/SumSum?gradients/classification_layers/dense0/dropout/div_grad/RealDivMgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients/classification_layers/dense0/dropout/div_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/div_grad/Sum=gradients/classification_layers/dense0/dropout/div_grad/Shape*'
_output_shapes
:џџџџџџџџџ
*
Tshape0*
T0

;gradients/classification_layers/dense0/dropout/div_grad/NegNeg!classification_layers/dense0/Relu*'
_output_shapes
:џџџџџџџџџ
*
T0
Я
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_1RealDiv;gradients/classification_layers/dense0/dropout/div_grad/Neg!classification_layers/Placeholder*
_output_shapes
:*
T0
е
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_2RealDivAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_1!classification_layers/Placeholder*
_output_shapes
:*
T0
њ
;gradients/classification_layers/dense0/dropout/div_grad/mulMulPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:

=gradients/classification_layers/dense0/dropout/div_grad/Sum_1Sum;gradients/classification_layers/dense0/dropout/div_grad/mulOgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
§
Agradients/classification_layers/dense0/dropout/div_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/div_grad/Sum_1?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
ж
Hgradients/classification_layers/dense0/dropout/div_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/div_grad/ReshapeB^gradients/classification_layers/dense0/dropout/div_grad/Reshape_1
ю
Pgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/div_grad/ReshapeI^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

х
Rgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/div_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape_1
ь
9gradients/classification_layers/dense0/Relu_grad/ReluGradReluGradPgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency!classification_layers/dense0/Relu*'
_output_shapes
:џџџџџџџџџ
*
T0
е
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ShapeShape@classification_layers/dense0/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
out_type0*
T0
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/SumSum9gradients/classification_layers/dense0/Relu_grad/ReluGradegradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ю
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape*
T0*'
_output_shapes
:џџџџџџџџџ
*
Tshape0
Р
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Sum_1Sum9gradients/classification_layers/dense0/Relu_grad/ReluGradggradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
_output_shapes
:
*
Tshape0

`gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1
Ю
hgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape
Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes
:
*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1*
T0
П
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
_output_shapes
:*
out_type0*
T0
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mulMulhgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency>classification_layers/dense0/batch_normalization/batchnorm/mul*'
_output_shapes
:џџџџџџџџџ
*
T0
ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/SumSumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mulegradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ
*
Tshape0*
T0
Є
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mul_1Mul*classification_layers/dense0/dense/BiasAddhgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ
*
T0
м
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Sum_1SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mul_1ggradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
_output_shapes
:
*
Tshape0

`gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ю
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
*
T0
Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:
*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1

Sgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:

Ugradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
щ
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/SumSumjgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1cgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape*
T0*
_output_shapes
:
*
Tshape0
э
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Sum_1Sumjgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1egradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
а
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/NegNegSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
П
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1ReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/NegUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:


^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*
_output_shapes
:
*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape
П
hgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1*
_output_shapes
:


Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ShapeConst*
valueB:
*
_output_shapes
:*
dtype0
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Љ
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mulMulhgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1>classification_layers/dense0/batch_normalization/batchnorm/mul*
_output_shapes
:
*
T0
ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/SumSumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mulegradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
С
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:

Ѕ
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mul_1Mul8classification_layers/dense0/batch_normalization/Squeezehgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1*
T0*
_output_shapes
:

м
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Sum_1SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mul_1ggradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0

`gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1
С
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape
Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:


Mgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
Ъ
Ogradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ReshapeReshapehgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyMgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/Shape*
Tshape0*
_output_shapes

:
*
T0

gradients/AddNAddNjgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes
:
*
T0*
N

Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:


Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ъ
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mulMulgradients/AddN;classification_layers/dense0/batch_normalization/gamma/read*
_output_shapes
:
*
T0
а
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/SumSumQgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mulcgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape*
T0*
_output_shapes
:
*
Tshape0
б
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mul_1Mul@classification_layers/dense0/batch_normalization/batchnorm/Rsqrtgradients/AddN*
_output_shapes
:
*
T0
ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Sum_1SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mul_1egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1ReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Sum_1Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0

^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_deps*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape*
_output_shapes
:
*
T0
П
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*
_output_shapes
:
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1
І
Qgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_likeConst*
dtype0*
_output_shapes

:
*
valueB
*    
о
Mgradients/classification_layers/dense0/batch_normalization/Select_grad/SelectSelect8classification_layers/dense0/batch_normalization/ReshapeOgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ReshapeQgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_like*
_output_shapes

:
*
T0
р
Ogradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1Select8classification_layers/dense0/batch_normalization/ReshapeQgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_likeOgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/Reshape*
_output_shapes

:
*
T0

Wgradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_depsNoOpN^gradients/classification_layers/dense0/batch_normalization/Select_grad/SelectP^gradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1

_gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependencyIdentityMgradients/classification_layers/dense0/batch_normalization/Select_grad/SelectX^gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/classification_layers/dense0/batch_normalization/Select_grad/Select*
_output_shapes

:

Ѕ
agradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependency_1IdentityOgradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1X^gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_deps*b
_classX
VTloc:@gradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1*
_output_shapes

:
*
T0
Е
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad@classification_layers/dense0/batch_normalization/batchnorm/Rsqrtfgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:


Pgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
У
Rgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ReshapeReshape_gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependencyPgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/Shape*
T0*
Tshape0*
_output_shapes
:


Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:

Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
и
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/SumSumYgradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradcgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape*
T0*
_output_shapes
:
*
Tshape0
м
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Sum_1SumYgradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradegradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Н
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1ReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Sum_1Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape*
_output_shapes
:

Л
hgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_deps*
T0*
_output_shapes
: *j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1
І
Ugradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
Ф
Wgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/ReshapeReshapeRgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ReshapeUgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Shape*
Tshape0*
_output_shapes

:
*
T0
 
Ogradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
Ь
Qgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ReshapeReshapefgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependencyOgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/Shape*
Tshape0*
_output_shapes

:
*
T0
Ѓ
Rgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
Ѕ
Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   
т
bgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ShapeTgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
д
Pgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/SumSumWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Reshapebgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
М
Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ReshapeReshapePgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/SumRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape*
T0*
_output_shapes

:
*
Tshape0
и
Rgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Sum_1SumWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Reshapedgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Т
Vgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1ReshapeRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Sum_1Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0

]gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_depsNoOpU^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ReshapeW^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1
Й
egradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependencyIdentityTgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape^^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_deps*
T0*
_output_shapes

:
*g
_class]
[Yloc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape
П
ggradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependency_1IdentityVgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1^^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1*
_output_shapes

:

Ј
Sgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_likeConst*
valueB
*    *
_output_shapes

:
*
dtype0
ц
Ogradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectSelect:classification_layers/dense0/batch_normalization/Reshape_1Qgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ReshapeSgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_like*
T0*
_output_shapes

:

ш
Qgradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1Select:classification_layers/dense0/batch_normalization/Reshape_1Sgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_likeQgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/Reshape*
_output_shapes

:
*
T0

Ygradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_depsNoOpP^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectR^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1
Ї
agradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependencyIdentityOgradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectZ^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_deps*
_output_shapes

:
*b
_classX
VTloc:@gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select*
T0
­
cgradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependency_1IdentityQgradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1Z^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1*
_output_shapes

:
*
T0

Rgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
Щ
Tgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ReshapeReshapeagradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependencyRgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:

Ј
Wgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   
   
Ъ
Ygradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/ReshapeReshapeTgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ReshapeWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:

Ї
Vgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ShapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Љ
Xgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
ю
fgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ShapeXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
о
Tgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/SumSumYgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Reshapefgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ш
Xgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ReshapeReshapeTgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/SumVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape*
_output_shapes

:
*
Tshape0*
T0
т
Vgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Sum_1SumYgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Reshapehgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ж
Tgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/NegNegVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Sum_1*
T0*
_output_shapes
:
Ь
Zgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1ReshapeTgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/NegXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0
Ё
agradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_depsNoOpY^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape[^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1
Щ
igradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependencyIdentityXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshapeb^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_deps*k
_classa
_]loc:@gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape*
_output_shapes

:
*
T0
Я
kgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1IdentityZgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1b^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_deps*m
_classc
a_loc:@gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1*
_output_shapes

:
*
T0
о
Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeShapeJclassification_layers/dense0/batch_normalization/moments/SquaredDifference*
out_type0*
_output_shapes
:*
T0

Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
І
Rgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/addAddQclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indicesSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Size*
T0*
_output_shapes
:
Ќ
Rgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/modFloorModRgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/addSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Size*
_output_shapes
:*
T0
 
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/rangeRangeZgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/startSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/SizeZgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/delta*

Tidx0*
_output_shapes
:

Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Г
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/FillFillVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_1Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill/value*
_output_shapes
:*
T0
љ
\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitchDynamicStitchTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/rangeRgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/modTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
N*
T0

Xgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ч
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/MaximumMaximum\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitchXgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0
Ж
Wgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordivFloorDivTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum*
T0*
_output_shapes
:
л
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ReshapeReshapeigradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
й
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/TileTileVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ReshapeWgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
р
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_2ShapeJclassification_layers/dense0/batch_normalization/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
Ї
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_3Const*
valueB"   
   *
dtype0*
_output_shapes
:

Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ч
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ProdProdVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_2Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
 
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
Ы
Ugradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Prod_1ProdVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_3Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
З
Xgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1MaximumUgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Prod_1Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1/y*
_output_shapes
: *
T0
Е
Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv_1FloorDivSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ProdXgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0
ц
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/CastCastYgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Н
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truedivRealDivSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/TileSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ


Tgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul/xConstl^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1*
_output_shapes
: *
dtype0*
valueB
 *   @

Rgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mulMulTgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul/xEclassification_layers/dense0/batch_normalization/moments/shifted_mean*
T0*
_output_shapes

:

Х
Tgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul_1Mulkgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1Rgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul*
_output_shapes

:
*
T0
Щ
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
_output_shapes
:*
out_type0*
T0
В
agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   

ogradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shapeagradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ў
`gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/scalarConstW^gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
а
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mulMul`gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/scalarVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*'
_output_shapes
:џџџџџџџџџ
*
T0
т
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/subSub*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradientW^gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ

ж
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1Mul]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

і
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/SumSum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1ogradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ь
agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/ReshapeReshape]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape*'
_output_shapes
:џџџџџџџџџ
*
Tshape0*
T0
њ
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_1Sum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1qgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
щ
cgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape_1Reshape_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_1agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
ђ
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/NegNegcgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:

Ж
jgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_depsNoOpb^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape^^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Neg
і
rgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyIdentityagradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshapek^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

ч
tgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1Identity]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Negk^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Neg*
_output_shapes

:

№
gradients/AddN_1AddNegradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependencyTgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul_1*g
_class]
[Yloc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape*
_output_shapes

:
*
T0*
N
ж
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ShapeShape<classification_layers/dense0/batch_normalization/moments/Sub*
_output_shapes
:*
out_type0*
T0

Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
И
Xgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/addAddWclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indicesYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size*
T0*
_output_shapes
:
О
Xgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/modFloorModXgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/addYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size*
_output_shapes
:*
T0
І
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Њ
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/rangeRange`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/startYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/delta*
_output_shapes
:*

Tidx0
Ё
_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Х
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/FillFill\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_1_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill/value*
_output_shapes
:*
T0

bgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitchDynamicStitchZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/rangeXgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/modZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ShapeYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ
 
^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
й
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/MaximumMaximumbgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitch^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0
Ш
]gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordivFloorDivZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum*
_output_shapes
:*
T0

\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ReshapeReshapegradients/AddN_1bgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
ы
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/TileTile\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Reshape]gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*

Tmultiples0
и
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_2Shape<classification_layers/dense0/batch_normalization/moments/Sub*
T0*
_output_shapes
:*
out_type0
­
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_3Const*
valueB"   
   *
dtype0*
_output_shapes
:
Є
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
й
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ProdProd\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_2Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
І
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
н
[gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod_1Prod\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_3\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1/yConst*
value	B :*
_output_shapes
: *
dtype0
Щ
^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1Maximum[gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod_1`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
Ч
_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv_1FloorDivYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1*
_output_shapes
: *
T0
ђ
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/CastCast_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Я
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivRealDivYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/TileYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ
*
T0
Л
Qgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
T0*
_output_shapes
:*
out_type0
Є
Sgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
п
agradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ShapeSgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
з
Ogradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/SumSum\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivagradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Т
Sgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ReshapeReshapeOgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/SumQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ
*
T0
л
Qgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Sum_1Sum\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivcgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ь
Ogradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/NegNegQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Sum_1*
_output_shapes
:*
T0
Н
Ugradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1ReshapeOgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/NegSgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0

\gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_depsNoOpT^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ReshapeV^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1
О
dgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependencyIdentitySgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape]^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*f
_class\
ZXloc:@gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape
Л
fgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency_1IdentityUgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1]^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_deps*
T0*
_output_shapes

:
*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1
ќ
gradients/AddN_2AddNggradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependency_1tgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1fgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency_1*
N*
T0*
_output_shapes

:
*i
_class_
][loc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1

gradients/AddN_3AddNhgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyrgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencydgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency*
T0*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

Ђ
Egradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes
:
*
T0*
data_formatNHWC
­
Jgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3F^gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad
л
Rgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3K^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

ѓ
Tgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*X
_classN
LJloc:@gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad*
T0
І
?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulMatMulRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency.classification_layers/dense0/dense/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

Agradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1MatMulFlatten/ReshapeRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	
*
transpose_a(*
T0
з
Igradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dense/MatMul_grad/MatMulB^gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1
ё
Qgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulJ^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*R
_classH
FDloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul*
T0
ю
Sgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1J^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	
*T
_classJ
HFloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1

beta1_power/initial_valueConst*
valueB
 *fff?*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
dtype0
­
beta1_power
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel
Ь
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

beta1_power/readIdentitybeta1_power*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0

beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wО?*<
_class2
0.loc:@classification_layers/dense0/dense/kernel
­
beta2_power
VariableV2*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
Ь
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
: 

beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel
е
@classification_layers/dense0/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB	
*    
т
.classification_layers/dense0/dense/kernel/Adam
VariableV2*
shared_name *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	

Т
5classification_layers/dense0/dense/kernel/Adam/AssignAssign.classification_layers/dense0/dense/kernel/Adam@classification_layers/dense0/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
:	

з
3classification_layers/dense0/dense/kernel/Adam/readIdentity.classification_layers/dense0/dense/kernel/Adam*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	

з
Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB	
*    *
_output_shapes
:	
*
dtype0
ф
0classification_layers/dense0/dense/kernel/Adam_1
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
shared_name *
_output_shapes
:	
*
shape:	

Ш
7classification_layers/dense0/dense/kernel/Adam_1/AssignAssign0classification_layers/dense0/dense/kernel/Adam_1Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
л
5classification_layers/dense0/dense/kernel/Adam_1/readIdentity0classification_layers/dense0/dense/kernel/Adam_1*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0
Ч
>classification_layers/dense0/dense/bias/Adam/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
_output_shapes
:
*
dtype0
д
,classification_layers/dense0/dense/bias/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias
Е
3classification_layers/dense0/dense/bias/Adam/AssignAssign,classification_layers/dense0/dense/bias/Adam>classification_layers/dense0/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
validate_shape(*
_output_shapes
:

Ь
1classification_layers/dense0/dense/bias/Adam/readIdentity,classification_layers/dense0/dense/bias/Adam*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0
Щ
@classification_layers/dense0/dense/bias/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
_output_shapes
:
*
dtype0
ж
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
Л
5classification_layers/dense0/dense/bias/Adam_1/AssignAssign.classification_layers/dense0/dense/bias/Adam_1@classification_layers/dense0/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias
а
3classification_layers/dense0/dense/bias/Adam_1/readIdentity.classification_layers/dense0/dense/bias/Adam_1*
T0*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias
у
Lclassification_layers/dense0/batch_normalization/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    
№
:classification_layers/dense0/batch_normalization/beta/Adam
VariableV2*
shared_name *H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
	container *
shape:
*
dtype0*
_output_shapes
:

э
Aclassification_layers/dense0/batch_normalization/beta/Adam/AssignAssign:classification_layers/dense0/batch_normalization/beta/AdamLclassification_layers/dense0/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta
і
?classification_layers/dense0/batch_normalization/beta/Adam/readIdentity:classification_layers/dense0/batch_normalization/beta/Adam*
T0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:

х
Nclassification_layers/dense0/batch_normalization/beta/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    *
_output_shapes
:
*
dtype0
ђ
<classification_layers/dense0/batch_normalization/beta/Adam_1
VariableV2*
	container *
dtype0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
shared_name *
_output_shapes
:
*
shape:

ѓ
Cclassification_layers/dense0/batch_normalization/beta/Adam_1/AssignAssign<classification_layers/dense0/batch_normalization/beta/Adam_1Nclassification_layers/dense0/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
validate_shape(*
_output_shapes
:

њ
Aclassification_layers/dense0/batch_normalization/beta/Adam_1/readIdentity<classification_layers/dense0/batch_normalization/beta/Adam_1*
T0*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta
х
Mclassification_layers/dense0/batch_normalization/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*    
ђ
;classification_layers/dense0/batch_normalization/gamma/Adam
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
shared_name 
ё
Bclassification_layers/dense0/batch_normalization/gamma/Adam/AssignAssign;classification_layers/dense0/batch_normalization/gamma/AdamMclassification_layers/dense0/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma
љ
@classification_layers/dense0/batch_normalization/gamma/Adam/readIdentity;classification_layers/dense0/batch_normalization/gamma/Adam*
T0*
_output_shapes
:
*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma
ч
Oclassification_layers/dense0/batch_normalization/gamma/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*    *
dtype0*
_output_shapes
:

є
=classification_layers/dense0/batch_normalization/gamma/Adam_1
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
shared_name 
ї
Dclassification_layers/dense0/batch_normalization/gamma/Adam_1/AssignAssign=classification_layers/dense0/batch_normalization/gamma/Adam_1Oclassification_layers/dense0/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
validate_shape(*
_output_shapes
:

§
Bclassification_layers/dense0/batch_normalization/gamma/Adam_1/readIdentity=classification_layers/dense0/batch_normalization/gamma/Adam_1*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:
*
T0
л
Dclassification_layers/dense_last/dense/kernel/Adam/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

ш
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
б
9classification_layers/dense_last/dense/kernel/Adam/AssignAssign2classification_layers/dense_last/dense/kernel/AdamDclassification_layers/dense_last/dense/kernel/Adam/Initializer/zeros*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
т
7classification_layers/dense_last/dense/kernel/Adam/readIdentity2classification_layers/dense_last/dense/kernel/Adam*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
н
Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

ъ
4classification_layers/dense_last/dense/kernel/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
з
;classification_layers/dense_last/dense/kernel/Adam_1/AssignAssign4classification_layers/dense_last/dense/kernel/Adam_1Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zeros*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
ц
9classification_layers/dense_last/dense/kernel/Adam_1/readIdentity4classification_layers/dense_last/dense/kernel/Adam_1*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
Я
Bclassification_layers/dense_last/dense/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
м
0classification_layers/dense_last/dense/bias/Adam
VariableV2*
shared_name *>
_class4
20loc:@classification_layers/dense_last/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Х
7classification_layers/dense_last/dense/bias/Adam/AssignAssign0classification_layers/dense_last/dense/bias/AdamBclassification_layers/dense_last/dense/bias/Adam/Initializer/zeros*
_output_shapes
:*
validate_shape(*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
T0*
use_locking(
и
5classification_layers/dense_last/dense/bias/Adam/readIdentity0classification_layers/dense_last/dense/bias/Adam*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:
б
Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
о
2classification_layers/dense_last/dense/bias/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@classification_layers/dense_last/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ы
9classification_layers/dense_last/dense/bias/Adam_1/AssignAssign2classification_layers/dense_last/dense/bias/Adam_1Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
validate_shape(*
_output_shapes
:
м
7classification_layers/dense_last/dense/bias/Adam_1/readIdentity2classification_layers/dense_last/dense/bias/Adam_1*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o:
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
_output_shapes
: *
dtype0

?Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam	ApplyAdam)classification_layers/dense0/dense/kernel.classification_layers/dense0/dense/kernel/Adam0classification_layers/dense0/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	
*
use_nesterov( *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0*
use_locking( 

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

х
KAdam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdam	ApplyAdam5classification_layers/dense0/batch_normalization/beta:classification_layers/dense0/batch_normalization/beta/Adam<classification_layers/dense0/batch_normalization/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonfgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
use_nesterov( *
_output_shapes
:

ь
LAdam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdam	ApplyAdam6classification_layers/dense0/batch_normalization/gamma;classification_layers/dense0/batch_normalization/gamma/Adam=classification_layers/dense0/batch_normalization/gamma/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonhgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_output_shapes
:
*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma
В
CAdam/update_classification_layers/dense_last/dense/kernel/ApplyAdam	ApplyAdam-classification_layers/dense_last/dense/kernel2classification_layers/dense_last/dense/kernel/Adam4classification_layers/dense_last/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonWgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
Ѕ
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
Е
Adam/mulMulbeta1_power/read
Adam/beta1@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0
Д
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
validate_shape(*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0*
use_locking( 
З

Adam/mul_1Mulbeta2_power/read
Adam/beta2@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0
И
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
: 
г
AdamNoOp@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
ћ	
initNoOp1^classification_layers/dense0/dense/kernel/Assign/^classification_layers/dense0/dense/bias/Assign=^classification_layers/dense0/batch_normalization/beta/Assign>^classification_layers/dense0/batch_normalization/gamma/AssignD^classification_layers/dense0/batch_normalization/moving_mean/AssignH^classification_layers/dense0/batch_normalization/moving_variance/Assign5^classification_layers/dense_last/dense/kernel/Assign3^classification_layers/dense_last/dense/bias/Assign^beta1_power/Assign^beta2_power/Assign6^classification_layers/dense0/dense/kernel/Adam/Assign8^classification_layers/dense0/dense/kernel/Adam_1/Assign4^classification_layers/dense0/dense/bias/Adam/Assign6^classification_layers/dense0/dense/bias/Adam_1/AssignB^classification_layers/dense0/batch_normalization/beta/Adam/AssignD^classification_layers/dense0/batch_normalization/beta/Adam_1/AssignC^classification_layers/dense0/batch_normalization/gamma/Adam/AssignE^classification_layers/dense0/batch_normalization/gamma/Adam_1/Assign:^classification_layers/dense_last/dense/kernel/Adam/Assign<^classification_layers/dense_last/dense/kernel/Adam_1/Assign8^classification_layers/dense_last/dense/bias/Adam/Assign:^classification_layers/dense_last/dense/bias/Adam_1/Assign"дIZ     Цћ(	­ ќЃ^жAJД	
ѓ#ж#
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
ы
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignSub
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
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

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
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

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
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
2	
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
-
Rsqrt
x"T
y"T"
Ttype:	
2
9
	RsqrtGrad
x"T
y"T
z"T"
Ttype:	
2
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
0
Square
x"T
y"T"
Ttype:
	2	
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

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
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.2.12v1.2.0-5-g435cdfcщ
|
Input/PlaceholderPlaceholder*+
_output_shapes
:џџџџџџџџџ *
dtype0* 
shape:џџџџџџџџџ 
u
Target/PlaceholderPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
g
"controll_normalization/PlaceholderPlaceholder*
_output_shapes
:*
shape:*
dtype0

^
Flatten/ShapeShapeInput/Placeholder*
T0*
out_type0*
_output_shapes
:
]
Flatten/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
\
Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

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
Flatten/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:

Flatten/Slice_1SliceFlatten/ShapeFlatten/Slice_1/beginFlatten/Slice_1/size*
_output_shapes
:*
T0*
Index0
W
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
r
Flatten/ProdProdFlatten/Slice_1Flatten/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
X
Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
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

Flatten/concatConcatV2Flatten/SliceFlatten/ExpandDimsFlatten/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
~
Flatten/ReshapeReshapeInput/PlaceholderFlatten/concat*
T0*(
_output_shapes
:џџџџџџџџџ*
Tshape0
f
!classification_layers/PlaceholderPlaceholder*
_output_shapes
:*
dtype0*
shape:
л
Lclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shapeConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB"   
   *
_output_shapes
:*
dtype0
Ю
Kclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/meanConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
а
Mclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddevConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Х
Vclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/shape*
T0*
_output_shapes
:	
*

seed *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
seed2 
р
Jclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulMulVclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/TruncatedNormalMclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel
Ю
Fclassification_layers/dense0/dense/kernel/Initializer/truncated_normalAddJclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mulKclassification_layers/dense0/dense/kernel/Initializer/truncated_normal/mean*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	

н
)classification_layers/dense0/dense/kernel
VariableV2*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	
*
shape:	
*
dtype0*
shared_name *
	container 
О
0classification_layers/dense0/dense/kernel/AssignAssign)classification_layers/dense0/dense/kernelFclassification_layers/dense0/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
:	

Э
.classification_layers/dense0/dense/kernel/readIdentity)classification_layers/dense0/dense/kernel*
T0*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel
Т
9classification_layers/dense0/dense/bias/Initializer/zerosConst*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    *
dtype0*
_output_shapes
:

Я
'classification_layers/dense0/dense/bias
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
І
.classification_layers/dense0/dense/bias/AssignAssign'classification_layers/dense0/dense/bias9classification_layers/dense0/dense/bias/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
validate_shape(*
_output_shapes
:

Т
,classification_layers/dense0/dense/bias/readIdentity'classification_layers/dense0/dense/bias*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:
*
T0
Ь
)classification_layers/dense0/dense/MatMulMatMulFlatten/Reshape.classification_layers/dense0/dense/kernel/read*
transpose_b( *'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
T0
з
*classification_layers/dense0/dense/BiasAddBiasAdd)classification_layers/dense0/dense/MatMul,classification_layers/dense0/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

о
Gclassification_layers/dense0/batch_normalization/beta/Initializer/zerosConst*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    *
dtype0*
_output_shapes
:

ы
5classification_layers/dense0/batch_normalization/beta
VariableV2*
	container *
dtype0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:
*
shape:
*
shared_name 
о
<classification_layers/dense0/batch_normalization/beta/AssignAssign5classification_layers/dense0/batch_normalization/betaGclassification_layers/dense0/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
validate_shape(*
_output_shapes
:

ь
:classification_layers/dense0/batch_normalization/beta/readIdentity5classification_layers/dense0/batch_normalization/beta*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:
*
T0
п
Gclassification_layers/dense0/batch_normalization/gamma/Initializer/onesConst*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*  ?*
_output_shapes
:
*
dtype0
э
6classification_layers/dense0/batch_normalization/gamma
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
dtype0*
	container 
с
=classification_layers/dense0/batch_normalization/gamma/AssignAssign6classification_layers/dense0/batch_normalization/gammaGclassification_layers/dense0/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
validate_shape(*
_output_shapes
:

я
;classification_layers/dense0/batch_normalization/gamma/readIdentity6classification_layers/dense0/batch_normalization/gamma*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:
*
T0
ь
Nclassification_layers/dense0/batch_normalization/moving_mean/Initializer/zerosConst*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
valueB
*    *
dtype0*
_output_shapes
:

љ
<classification_layers/dense0/batch_normalization/moving_mean
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean
њ
Cclassification_layers/dense0/batch_normalization/moving_mean/AssignAssign<classification_layers/dense0/batch_normalization/moving_meanNclassification_layers/dense0/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:


Aclassification_layers/dense0/batch_normalization/moving_mean/readIdentity<classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:
*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
T0
ѓ
Qclassification_layers/dense0/batch_normalization/moving_variance/Initializer/onesConst*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
valueB
*  ?*
dtype0*
_output_shapes
:


@classification_layers/dense0/batch_normalization/moving_variance
VariableV2*
shape:
*
_output_shapes
:
*
shared_name *S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
dtype0*
	container 

Gclassification_layers/dense0/batch_normalization/moving_variance/AssignAssign@classification_layers/dense0/batch_normalization/moving_varianceQclassification_layers/dense0/batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:
*
validate_shape(*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
T0*
use_locking(

Eclassification_layers/dense0/batch_normalization/moving_variance/readIdentity@classification_layers/dense0/batch_normalization/moving_variance*
T0*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:


Oclassification_layers/dense0/batch_normalization/moments/Mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:

=classification_layers/dense0/batch_normalization/moments/MeanMean*classification_layers/dense0/dense/BiasAddOclassification_layers/dense0/batch_normalization/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes

:

Н
Eclassification_layers/dense0/batch_normalization/moments/StopGradientStopGradient=classification_layers/dense0/batch_normalization/moments/Mean*
_output_shapes

:
*
T0
ш
<classification_layers/dense0/batch_normalization/moments/SubSub*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradient*
T0*'
_output_shapes
:џџџџџџџџџ

Ё
Wclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Њ
Eclassification_layers/dense0/batch_normalization/moments/shifted_meanMean<classification_layers/dense0/batch_normalization/moments/SubWclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indices*
_output_shapes

:
*
T0*
	keep_dims(*

Tidx0

Jclassification_layers/dense0/batch_normalization/moments/SquaredDifferenceSquaredDifference*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradient*'
_output_shapes
:џџџџџџџџџ
*
T0

Qclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
Ќ
?classification_layers/dense0/batch_normalization/moments/Mean_1MeanJclassification_layers/dense0/batch_normalization/moments/SquaredDifferenceQclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes

:

Й
?classification_layers/dense0/batch_normalization/moments/SquareSquareEclassification_layers/dense0/batch_normalization/moments/shifted_mean*
_output_shapes

:
*
T0
ѓ
Aclassification_layers/dense0/batch_normalization/moments/varianceSub?classification_layers/dense0/batch_normalization/moments/Mean_1?classification_layers/dense0/batch_normalization/moments/Square*
T0*
_output_shapes

:

ћ
=classification_layers/dense0/batch_normalization/moments/meanAddEclassification_layers/dense0/batch_normalization/moments/shifted_meanEclassification_layers/dense0/batch_normalization/moments/StopGradient*
T0*
_output_shapes

:

Ц
@classification_layers/dense0/batch_normalization/moments/SqueezeSqueeze=classification_layers/dense0/batch_normalization/moments/mean*
_output_shapes
:
*
T0*
squeeze_dims
 
Ь
Bclassification_layers/dense0/batch_normalization/moments/Squeeze_1SqueezeAclassification_layers/dense0/batch_normalization/moments/variance*
squeeze_dims
 *
_output_shapes
:
*
T0

?classification_layers/dense0/batch_normalization/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

;classification_layers/dense0/batch_normalization/ExpandDims
ExpandDims@classification_layers/dense0/batch_normalization/moments/Squeeze?classification_layers/dense0/batch_normalization/ExpandDims/dim*

Tdim0*
_output_shapes

:
*
T0

Aclassification_layers/dense0/batch_normalization/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 

=classification_layers/dense0/batch_normalization/ExpandDims_1
ExpandDimsAclassification_layers/dense0/batch_normalization/moving_mean/readAclassification_layers/dense0/batch_normalization/ExpandDims_1/dim*
T0*
_output_shapes

:
*

Tdim0

>classification_layers/dense0/batch_normalization/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
к
8classification_layers/dense0/batch_normalization/ReshapeReshape"controll_normalization/Placeholder>classification_layers/dense0/batch_normalization/Reshape/shape*
T0
*
_output_shapes
:*
Tshape0
 
7classification_layers/dense0/batch_normalization/SelectSelect8classification_layers/dense0/batch_normalization/Reshape;classification_layers/dense0/batch_normalization/ExpandDims=classification_layers/dense0/batch_normalization/ExpandDims_1*
T0*
_output_shapes

:

И
8classification_layers/dense0/batch_normalization/SqueezeSqueeze7classification_layers/dense0/batch_normalization/Select*
_output_shapes
:
*
T0*
squeeze_dims
 

Aclassification_layers/dense0/batch_normalization/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 

=classification_layers/dense0/batch_normalization/ExpandDims_2
ExpandDimsBclassification_layers/dense0/batch_normalization/moments/Squeeze_1Aclassification_layers/dense0/batch_normalization/ExpandDims_2/dim*
_output_shapes

:
*
T0*

Tdim0

Aclassification_layers/dense0/batch_normalization/ExpandDims_3/dimConst*
value	B : *
_output_shapes
: *
dtype0

=classification_layers/dense0/batch_normalization/ExpandDims_3
ExpandDimsEclassification_layers/dense0/batch_normalization/moving_variance/readAclassification_layers/dense0/batch_normalization/ExpandDims_3/dim*
_output_shapes

:
*
T0*

Tdim0

@classification_layers/dense0/batch_normalization/Reshape_1/shapeConst*
valueB:*
_output_shapes
:*
dtype0
о
:classification_layers/dense0/batch_normalization/Reshape_1Reshape"controll_normalization/Placeholder@classification_layers/dense0/batch_normalization/Reshape_1/shape*
T0
*
_output_shapes
:*
Tshape0
І
9classification_layers/dense0/batch_normalization/Select_1Select:classification_layers/dense0/batch_normalization/Reshape_1=classification_layers/dense0/batch_normalization/ExpandDims_2=classification_layers/dense0/batch_normalization/ExpandDims_3*
_output_shapes

:
*
T0
М
:classification_layers/dense0/batch_normalization/Squeeze_1Squeeze9classification_layers/dense0/batch_normalization/Select_1*
squeeze_dims
 *
_output_shapes
:
*
T0

Cclassification_layers/dense0/batch_normalization/ExpandDims_4/inputConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?

Aclassification_layers/dense0/batch_normalization/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 

=classification_layers/dense0/batch_normalization/ExpandDims_4
ExpandDimsCclassification_layers/dense0/batch_normalization/ExpandDims_4/inputAclassification_layers/dense0/batch_normalization/ExpandDims_4/dim*
T0*
_output_shapes
:*

Tdim0

Cclassification_layers/dense0/batch_normalization/ExpandDims_5/inputConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

Aclassification_layers/dense0/batch_normalization/ExpandDims_5/dimConst*
value	B : *
_output_shapes
: *
dtype0

=classification_layers/dense0/batch_normalization/ExpandDims_5
ExpandDimsCclassification_layers/dense0/batch_normalization/ExpandDims_5/inputAclassification_layers/dense0/batch_normalization/ExpandDims_5/dim*
_output_shapes
:*
T0*

Tdim0

@classification_layers/dense0/batch_normalization/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
о
:classification_layers/dense0/batch_normalization/Reshape_2Reshape"controll_normalization/Placeholder@classification_layers/dense0/batch_normalization/Reshape_2/shape*
T0
*
Tshape0*
_output_shapes
:
Ђ
9classification_layers/dense0/batch_normalization/Select_2Select:classification_layers/dense0/batch_normalization/Reshape_2=classification_layers/dense0/batch_normalization/ExpandDims_4=classification_layers/dense0/batch_normalization/ExpandDims_5*
_output_shapes
:*
T0
И
:classification_layers/dense0/batch_normalization/Squeeze_2Squeeze9classification_layers/dense0/batch_normalization/Select_2*
T0*
_output_shapes
: *
squeeze_dims
 
м
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub/xConst*
valueB
 *  ?*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
dtype0*
_output_shapes
: 
С
Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/subSubFclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub/x:classification_layers/dense0/batch_normalization/Squeeze_2*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
: 
Р
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub_1SubAclassification_layers/dense0/batch_normalization/moving_mean/read8classification_layers/dense0/batch_normalization/Squeeze*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:

Я
Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/mulMulFclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub_1Dclassification_layers/dense0/batch_normalization/AssignMovingAvg/sub*
T0*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
_output_shapes
:

к
@classification_layers/dense0/batch_normalization/AssignMovingAvg	AssignSub<classification_layers/dense0/batch_normalization/moving_meanDclassification_layers/dense0/batch_normalization/AssignMovingAvg/mul*
_output_shapes
:
*O
_classE
CAloc:@classification_layers/dense0/batch_normalization/moving_mean*
T0*
use_locking( 
т
Hclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub/xConst*
valueB
 *  ?*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
Щ
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/subSubHclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub/x:classification_layers/dense0/batch_normalization/Squeeze_2*
T0*
_output_shapes
: *S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance
Ь
Hclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub_1SubEclassification_layers/dense0/batch_normalization/moving_variance/read:classification_layers/dense0/batch_normalization/Squeeze_1*
_output_shapes
:
*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
T0
й
Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/mulMulHclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub_1Fclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/sub*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:
*
T0
ц
Bclassification_layers/dense0/batch_normalization/AssignMovingAvg_1	AssignSub@classification_layers/dense0/batch_normalization/moving_varianceFclassification_layers/dense0/batch_normalization/AssignMovingAvg_1/mul*S
_classI
GEloc:@classification_layers/dense0/batch_normalization/moving_variance*
_output_shapes
:
*
T0*
use_locking( 

@classification_layers/dense0/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:
ш
>classification_layers/dense0/batch_normalization/batchnorm/addAdd:classification_layers/dense0/batch_normalization/Squeeze_1@classification_layers/dense0/batch_normalization/batchnorm/add/y*
T0*
_output_shapes
:

Ў
@classification_layers/dense0/batch_normalization/batchnorm/RsqrtRsqrt>classification_layers/dense0/batch_normalization/batchnorm/add*
T0*
_output_shapes
:

щ
>classification_layers/dense0/batch_normalization/batchnorm/mulMul@classification_layers/dense0/batch_normalization/batchnorm/Rsqrt;classification_layers/dense0/batch_normalization/gamma/read*
T0*
_output_shapes
:

х
@classification_layers/dense0/batch_normalization/batchnorm/mul_1Mul*classification_layers/dense0/dense/BiasAdd>classification_layers/dense0/batch_normalization/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ

ц
@classification_layers/dense0/batch_normalization/batchnorm/mul_2Mul8classification_layers/dense0/batch_normalization/Squeeze>classification_layers/dense0/batch_normalization/batchnorm/mul*
T0*
_output_shapes
:

ш
>classification_layers/dense0/batch_normalization/batchnorm/subSub:classification_layers/dense0/batch_normalization/beta/read@classification_layers/dense0/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes
:

ћ
@classification_layers/dense0/batch_normalization/batchnorm/add_1Add@classification_layers/dense0/batch_normalization/batchnorm/mul_1>classification_layers/dense0/batch_normalization/batchnorm/sub*
T0*'
_output_shapes
:џџџџџџџџџ


!classification_layers/dense0/ReluRelu@classification_layers/dense0/batch_normalization/batchnorm/add_1*
T0*'
_output_shapes
:џџџџџџџџџ


*classification_layers/dense0/dropout/ShapeShape!classification_layers/dense0/Relu*
out_type0*
_output_shapes
:*
T0
|
7classification_layers/dense0/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
|
7classification_layers/dense0/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ж
Aclassification_layers/dense0/dropout/random_uniform/RandomUniformRandomUniform*classification_layers/dense0/dropout/Shape*

seed *
T0*
dtype0*'
_output_shapes
:џџџџџџџџџ
*
seed2 
б
7classification_layers/dense0/dropout/random_uniform/subSub7classification_layers/dense0/dropout/random_uniform/max7classification_layers/dense0/dropout/random_uniform/min*
T0*
_output_shapes
: 
ь
7classification_layers/dense0/dropout/random_uniform/mulMulAclassification_layers/dense0/dropout/random_uniform/RandomUniform7classification_layers/dense0/dropout/random_uniform/sub*
T0*'
_output_shapes
:џџџџџџџџџ

о
3classification_layers/dense0/dropout/random_uniformAdd7classification_layers/dense0/dropout/random_uniform/mul7classification_layers/dense0/dropout/random_uniform/min*
T0*'
_output_shapes
:џџџџџџџџџ

Њ
(classification_layers/dense0/dropout/addAdd!classification_layers/Placeholder3classification_layers/dense0/dropout/random_uniform*
_output_shapes
:*
T0

*classification_layers/dense0/dropout/FloorFloor(classification_layers/dense0/dropout/add*
_output_shapes
:*
T0

(classification_layers/dense0/dropout/divRealDiv!classification_layers/dense0/Relu!classification_layers/Placeholder*
T0*
_output_shapes
:
З
(classification_layers/dense0/dropout/mulMul(classification_layers/dense0/dropout/div*classification_layers/dense0/dropout/Floor*'
_output_shapes
:џџџџџџџџџ
*
T0
у
Pclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shapeConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB"
      *
_output_shapes
:*
dtype0
ж
Oclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/meanConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
и
Qclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddevConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
 *  ?*
_output_shapes
: *
dtype0
а
Zclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalPclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
я
Nclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulMulZclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/TruncatedNormalQclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/stddev*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
н
Jclassification_layers/dense_last/dense/kernel/Initializer/truncated_normalAddNclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mulOclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal/mean*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
T0
у
-classification_layers/dense_last/dense/kernel
VariableV2*
	container *
dtype0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
shared_name *
_output_shapes

:
*
shape
:

Э
4classification_layers/dense_last/dense/kernel/AssignAssign-classification_layers/dense_last/dense/kernelJclassification_layers/dense_last/dense/kernel/Initializer/truncated_normal*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
и
2classification_layers/dense_last/dense/kernel/readIdentity-classification_layers/dense_last/dense/kernel*
T0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:

Ъ
=classification_layers/dense_last/dense/bias/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
з
+classification_layers/dense_last/dense/bias
VariableV2*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ж
2classification_layers/dense_last/dense/bias/AssignAssign+classification_layers/dense_last/dense/bias=classification_layers/dense_last/dense/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias
Ю
0classification_layers/dense_last/dense/bias/readIdentity+classification_layers/dense_last/dense/bias*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
T0
э
-classification_layers/dense_last/dense/MatMulMatMul(classification_layers/dense0/dropout/mul2classification_layers/dense_last/dense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
у
.classification_layers/dense_last/dense/BiasAddBiasAdd-classification_layers/dense_last/dense/MatMul0classification_layers/dense_last/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

classification_layers/SoftmaxSoftmax.classification_layers/dense_last/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
n
)Evaluation_layers/clip_by_value/Minimum/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ў
'Evaluation_layers/clip_by_value/MinimumMinimumclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*'
_output_shapes
:џџџџџџџџџ*
T0
f
!Evaluation_layers/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *џцл.
Ј
Evaluation_layers/clip_by_valueMaximum'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
Evaluation_layers/LogLogEvaluation_layers/clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
y
Evaluation_layers/mulMulTarget/PlaceholderEvaluation_layers/Log*'
_output_shapes
:џџџџџџџџџ*
T0
q
'Evaluation_layers/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Ї
Evaluation_layers/SumSumEvaluation_layers/mul'Evaluation_layers/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
a
Evaluation_layers/NegNegEvaluation_layers/Sum*
T0*#
_output_shapes
:џџџџџџџџџ
a
Evaluation_layers/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

Evaluation_layers/MeanMeanEvaluation_layers/NegEvaluation_layers/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
d
"Evaluation_layers/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :

Evaluation_layers/ArgMaxArgMaxclassification_layers/Softmax"Evaluation_layers/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
f
$Evaluation_layers/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :

Evaluation_layers/ArgMax_1ArgMaxTarget/Placeholder$Evaluation_layers/ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

Evaluation_layers/EqualEqualEvaluation_layers/ArgMaxEvaluation_layers/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
|
Evaluation_layers/accracy/CastCastEvaluation_layers/Equal*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
i
Evaluation_layers/accracy/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ѕ
Evaluation_layers/accracy/MeanMeanEvaluation_layers/accracy/CastEvaluation_layers/accracy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
z
Evaluation_layers/accuracy/tagsConst*
dtype0*
_output_shapes
: *+
value"B  BEvaluation_layers/accuracy

Evaluation_layers/accuracyScalarSummaryEvaluation_layers/accuracy/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
r
Evaluation_layers/loss/tagsConst*
_output_shapes
: *
dtype0*'
valueB BEvaluation_layers/loss
}
Evaluation_layers/lossScalarSummaryEvaluation_layers/loss/tagsEvaluation_layers/Mean*
T0*
_output_shapes
: 
~
!Evaluation_layers/accuracy_1/tagsConst*-
value$B" BEvaluation_layers/accuracy_1*
dtype0*
_output_shapes
: 

Evaluation_layers/accuracy_1ScalarSummary!Evaluation_layers/accuracy_1/tagsEvaluation_layers/accracy/Mean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
}
3gradients/Evaluation_layers/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
А
-gradients/Evaluation_layers/Mean_grad/ReshapeReshapegradients/Fill3gradients/Evaluation_layers/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

+gradients/Evaluation_layers/Mean_grad/ShapeShapeEvaluation_layers/Neg*
_output_shapes
:*
out_type0*
T0
Ю
*gradients/Evaluation_layers/Mean_grad/TileTile-gradients/Evaluation_layers/Mean_grad/Reshape+gradients/Evaluation_layers/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0

-gradients/Evaluation_layers/Mean_grad/Shape_1ShapeEvaluation_layers/Neg*
T0*
_output_shapes
:*
out_type0
p
-gradients/Evaluation_layers/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
u
+gradients/Evaluation_layers/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ь
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
а
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
Ж
.gradients/Evaluation_layers/Mean_grad/floordivFloorDiv*gradients/Evaluation_layers/Mean_grad/Prod-gradients/Evaluation_layers/Mean_grad/Maximum*
T0*
_output_shapes
: 

*gradients/Evaluation_layers/Mean_grad/CastCast.gradients/Evaluation_layers/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
О
-gradients/Evaluation_layers/Mean_grad/truedivRealDiv*gradients/Evaluation_layers/Mean_grad/Tile*gradients/Evaluation_layers/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

(gradients/Evaluation_layers/Neg_grad/NegNeg-gradients/Evaluation_layers/Mean_grad/truediv*#
_output_shapes
:џџџџџџџџџ*
T0

*gradients/Evaluation_layers/Sum_grad/ShapeShapeEvaluation_layers/mul*
T0*
_output_shapes
:*
out_type0
k
)gradients/Evaluation_layers/Sum_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
(gradients/Evaluation_layers/Sum_grad/addAdd'Evaluation_layers/Sum/reduction_indices)gradients/Evaluation_layers/Sum_grad/Size*
_output_shapes
:*
T0
Ў
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
value	B :*
dtype0*
_output_shapes
: 
ъ
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
Е
)gradients/Evaluation_layers/Sum_grad/FillFill,gradients/Evaluation_layers/Sum_grad/Shape_1/gradients/Evaluation_layers/Sum_grad/Fill/value*
_output_shapes
:*
T0
Ї
2gradients/Evaluation_layers/Sum_grad/DynamicStitchDynamicStitch*gradients/Evaluation_layers/Sum_grad/range(gradients/Evaluation_layers/Sum_grad/mod*gradients/Evaluation_layers/Sum_grad/Shape)gradients/Evaluation_layers/Sum_grad/Fill*
N*
T0*#
_output_shapes
:џџџџџџџџџ
p
.gradients/Evaluation_layers/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Щ
,gradients/Evaluation_layers/Sum_grad/MaximumMaximum2gradients/Evaluation_layers/Sum_grad/DynamicStitch.gradients/Evaluation_layers/Sum_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ
И
-gradients/Evaluation_layers/Sum_grad/floordivFloorDiv*gradients/Evaluation_layers/Sum_grad/Shape,gradients/Evaluation_layers/Sum_grad/Maximum*
T0*
_output_shapes
:
Ц
,gradients/Evaluation_layers/Sum_grad/ReshapeReshape(gradients/Evaluation_layers/Neg_grad/Neg2gradients/Evaluation_layers/Sum_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
в
)gradients/Evaluation_layers/Sum_grad/TileTile,gradients/Evaluation_layers/Sum_grad/Reshape-gradients/Evaluation_layers/Sum_grad/floordiv*'
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0
|
*gradients/Evaluation_layers/mul_grad/ShapeShapeTarget/Placeholder*
T0*
_output_shapes
:*
out_type0

,gradients/Evaluation_layers/mul_grad/Shape_1ShapeEvaluation_layers/Log*
T0*
_output_shapes
:*
out_type0
ъ
:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/Evaluation_layers/mul_grad/Shape,gradients/Evaluation_layers/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѓ
(gradients/Evaluation_layers/mul_grad/mulMul)gradients/Evaluation_layers/Sum_grad/TileEvaluation_layers/Log*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients/Evaluation_layers/mul_grad/SumSum(gradients/Evaluation_layers/mul_grad/mul:gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
,gradients/Evaluation_layers/mul_grad/ReshapeReshape(gradients/Evaluation_layers/mul_grad/Sum*gradients/Evaluation_layers/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
Ђ
*gradients/Evaluation_layers/mul_grad/mul_1MulTarget/Placeholder)gradients/Evaluation_layers/Sum_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0
л
*gradients/Evaluation_layers/mul_grad/Sum_1Sum*gradients/Evaluation_layers/mul_grad/mul_1<gradients/Evaluation_layers/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
г
.gradients/Evaluation_layers/mul_grad/Reshape_1Reshape*gradients/Evaluation_layers/mul_grad/Sum_1,gradients/Evaluation_layers/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5gradients/Evaluation_layers/mul_grad/tuple/group_depsNoOp-^gradients/Evaluation_layers/mul_grad/Reshape/^gradients/Evaluation_layers/mul_grad/Reshape_1
Ђ
=gradients/Evaluation_layers/mul_grad/tuple/control_dependencyIdentity,gradients/Evaluation_layers/mul_grad/Reshape6^gradients/Evaluation_layers/mul_grad/tuple/group_deps*?
_class5
31loc:@gradients/Evaluation_layers/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
Ј
?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1Identity.gradients/Evaluation_layers/mul_grad/Reshape_16^gradients/Evaluation_layers/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*A
_class7
53loc:@gradients/Evaluation_layers/mul_grad/Reshape_1*
T0
в
/gradients/Evaluation_layers/Log_grad/Reciprocal
ReciprocalEvaluation_layers/clip_by_value@^gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
г
(gradients/Evaluation_layers/Log_grad/mulMul?gradients/Evaluation_layers/mul_grad/tuple/control_dependency_1/gradients/Evaluation_layers/Log_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0

4gradients/Evaluation_layers/clip_by_value_grad/ShapeShape'Evaluation_layers/clip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0
y
6gradients/Evaluation_layers/clip_by_value_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0

6gradients/Evaluation_layers/clip_by_value_grad/Shape_2Shape(gradients/Evaluation_layers/Log_grad/mul*
out_type0*
_output_shapes
:*
T0

:gradients/Evaluation_layers/clip_by_value_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
т
4gradients/Evaluation_layers/clip_by_value_grad/zerosFill6gradients/Evaluation_layers/clip_by_value_grad/Shape_2:gradients/Evaluation_layers/clip_by_value_grad/zeros/Const*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqualGreaterEqual'Evaluation_layers/clip_by_value/Minimum!Evaluation_layers/clip_by_value/y*
T0*'
_output_shapes
:џџџџџџџџџ

Dgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/Evaluation_layers/clip_by_value_grad/Shape6gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

5gradients/Evaluation_layers/clip_by_value_grad/SelectSelect;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
­
9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot
LogicalNot;gradients/Evaluation_layers/clip_by_value_grad/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

7gradients/Evaluation_layers/clip_by_value_grad/Select_1Select9gradients/Evaluation_layers/clip_by_value_grad/LogicalNot(gradients/Evaluation_layers/Log_grad/mul4gradients/Evaluation_layers/clip_by_value_grad/zeros*'
_output_shapes
:џџџџџџџџџ*
T0
і
2gradients/Evaluation_layers/clip_by_value_grad/SumSum5gradients/Evaluation_layers/clip_by_value_grad/SelectDgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ы
6gradients/Evaluation_layers/clip_by_value_grad/ReshapeReshape2gradients/Evaluation_layers/clip_by_value_grad/Sum4gradients/Evaluation_layers/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ќ
4gradients/Evaluation_layers/clip_by_value_grad/Sum_1Sum7gradients/Evaluation_layers/clip_by_value_grad/Select_1Fgradients/Evaluation_layers/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
р
8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1Reshape4gradients/Evaluation_layers/clip_by_value_grad/Sum_16gradients/Evaluation_layers/clip_by_value_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Л
?gradients/Evaluation_layers/clip_by_value_grad/tuple/group_depsNoOp7^gradients/Evaluation_layers/clip_by_value_grad/Reshape9^gradients/Evaluation_layers/clip_by_value_grad/Reshape_1
Ъ
Ggradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependencyIdentity6gradients/Evaluation_layers/clip_by_value_grad/Reshape@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
П
Igradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency_1Identity8gradients/Evaluation_layers/clip_by_value_grad/Reshape_1@^gradients/Evaluation_layers/clip_by_value_grad/tuple/group_deps*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/Evaluation_layers/clip_by_value_grad/Reshape_1

<gradients/Evaluation_layers/clip_by_value/Minimum_grad/ShapeShapeclassification_layers/Softmax*
T0*
out_type0*
_output_shapes
:

>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Х
>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2ShapeGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
њ
<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zerosFill>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_2Bgradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual	LessEqualclassification_layers/Softmax)Evaluation_layers/clip_by_value/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
 
Lgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Т
=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectSelect@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqualGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ
К
Agradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNot
LogicalNot@gradients/Evaluation_layers/clip_by_value/Minimum_grad/LessEqual*'
_output_shapes
:џџџџџџџџџ
Х
?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1SelectAgradients/Evaluation_layers/clip_by_value/Minimum_grad/LogicalNotGgradients/Evaluation_layers/clip_by_value_grad/tuple/control_dependency<gradients/Evaluation_layers/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:џџџџџџџџџ

:gradients/Evaluation_layers/clip_by_value/Minimum_grad/SumSum=gradients/Evaluation_layers/clip_by_value/Minimum_grad/SelectLgradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeReshape:gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1Sum?gradients/Evaluation_layers/clip_by_value/Minimum_grad/Select_1Ngradients/Evaluation_layers/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ј
@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1Reshape<gradients/Evaluation_layers/clip_by_value/Minimum_grad/Sum_1>gradients/Evaluation_layers/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
г
Ggradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_depsNoOp?^gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeA^gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1
ъ
Ogradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity>gradients/Evaluation_layers/clip_by_value/Minimum_grad/ReshapeH^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
п
Qgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1H^gradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Evaluation_layers/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
й
0gradients/classification_layers/Softmax_grad/mulMulOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependencyclassification_layers/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ

Bgradients/classification_layers/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
ј
0gradients/classification_layers/Softmax_grad/SumSum0gradients/classification_layers/Softmax_grad/mulBgradients/classification_layers/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

:gradients/classification_layers/Softmax_grad/Reshape/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
э
4gradients/classification_layers/Softmax_grad/ReshapeReshape0gradients/classification_layers/Softmax_grad/Sum:gradients/classification_layers/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
№
0gradients/classification_layers/Softmax_grad/subSubOgradients/Evaluation_layers/clip_by_value/Minimum_grad/tuple/control_dependency4gradients/classification_layers/Softmax_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
М
2gradients/classification_layers/Softmax_grad/mul_1Mul0gradients/classification_layers/Softmax_grad/subclassification_layers/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ш
Igradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/classification_layers/Softmax_grad/mul_1*
_output_shapes
:*
T0*
data_formatNHWC
з
Ngradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/classification_layers/Softmax_grad/mul_1J^gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad
р
Vgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/classification_layers/Softmax_grad/mul_1O^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/classification_layers/Softmax_grad/mul_1*'
_output_shapes
:џџџџџџџџџ*
T0

Xgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGradO^gradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*\
_classR
PNloc:@gradients/classification_layers/dense_last/dense/BiasAdd_grad/BiasAddGrad*
T0
Б
Cgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulMatMulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency2classification_layers/dense_last/dense/kernel/read*
transpose_b(*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
T0
 
Egradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1MatMul(classification_layers/dense0/dropout/mulVgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
у
Mgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_depsNoOpD^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMulF^gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1

Ugradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependencyIdentityCgradients/classification_layers/dense_last/dense/MatMul_grad/MatMulN^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ

§
Wgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1N^gradients/classification_layers/dense_last/dense/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/classification_layers/dense_last/dense/MatMul_grad/MatMul_1*
_output_shapes

:

Ў
=gradients/classification_layers/dense0/dropout/mul_grad/ShapeShape(classification_layers/dense0/dropout/div*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
В
?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1Shape*classification_layers/dense0/dropout/Floor*#
_output_shapes
:џџџџџџџџџ*
out_type0*
T0
Ѓ
Mgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/mul_grad/Shape?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ш
;gradients/classification_layers/dense0/dropout/mul_grad/mulMulUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*classification_layers/dense0/dropout/Floor*
T0*
_output_shapes
:

;gradients/classification_layers/dense0/dropout/mul_grad/SumSum;gradients/classification_layers/dense0/dropout/mul_grad/mulMgradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/mul_grad/Sum=gradients/classification_layers/dense0/dropout/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0
ш
=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Mul(classification_layers/dense0/dropout/divUgradients/classification_layers/dense_last/dense/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0

=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1Sum=gradients/classification_layers/dense0/dropout/mul_grad/mul_1Ogradients/classification_layers/dense0/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
§
Agradients/classification_layers/dense0/dropout/mul_grad/Reshape_1Reshape=gradients/classification_layers/dense0/dropout/mul_grad/Sum_1?gradients/classification_layers/dense0/dropout/mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
ж
Hgradients/classification_layers/dense0/dropout/mul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/mul_grad/ReshapeB^gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1
п
Pgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/mul_grad/ReshapeI^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape
х
Rgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/mul_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0

=gradients/classification_layers/dense0/dropout/div_grad/ShapeShape!classification_layers/dense0/Relu*
T0*
_output_shapes
:*
out_type0
Љ
?gradients/classification_layers/dense0/dropout/div_grad/Shape_1Shape!classification_layers/Placeholder*
T0*#
_output_shapes
:џџџџџџџџџ*
out_type0
Ѓ
Mgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/classification_layers/dense0/dropout/div_grad/Shape?gradients/classification_layers/dense0/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
т
?gradients/classification_layers/dense0/dropout/div_grad/RealDivRealDivPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependency!classification_layers/Placeholder*
T0*
_output_shapes
:

;gradients/classification_layers/dense0/dropout/div_grad/SumSum?gradients/classification_layers/dense0/dropout/div_grad/RealDivMgradients/classification_layers/dense0/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

?gradients/classification_layers/dense0/dropout/div_grad/ReshapeReshape;gradients/classification_layers/dense0/dropout/div_grad/Sum=gradients/classification_layers/dense0/dropout/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ


;gradients/classification_layers/dense0/dropout/div_grad/NegNeg!classification_layers/dense0/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

Я
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_1RealDiv;gradients/classification_layers/dense0/dropout/div_grad/Neg!classification_layers/Placeholder*
T0*
_output_shapes
:
е
Agradients/classification_layers/dense0/dropout/div_grad/RealDiv_2RealDivAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_1!classification_layers/Placeholder*
_output_shapes
:*
T0
њ
;gradients/classification_layers/dense0/dropout/div_grad/mulMulPgradients/classification_layers/dense0/dropout/mul_grad/tuple/control_dependencyAgradients/classification_layers/dense0/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0

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
ж
Hgradients/classification_layers/dense0/dropout/div_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dropout/div_grad/ReshapeB^gradients/classification_layers/dense0/dropout/div_grad/Reshape_1
ю
Pgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dropout/div_grad/ReshapeI^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ
*R
_classH
FDloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape*
T0
х
Rgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dropout/div_grad/Reshape_1I^gradients/classification_layers/dense0/dropout/div_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/classification_layers/dense0/dropout/div_grad/Reshape_1*
_output_shapes
:
ь
9gradients/classification_layers/dense0/Relu_grad/ReluGradReluGradPgradients/classification_layers/dense0/dropout/div_grad/tuple/control_dependency!classification_layers/dense0/Relu*
T0*'
_output_shapes
:џџџџџџџџџ

е
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ShapeShape@classification_layers/dense0/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/SumSum9gradients/classification_layers/dense0/Relu_grad/ReluGradegradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

Р
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Sum_1Sum9gradients/classification_layers/dense0/Relu_grad/ReluGradggradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:


`gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1
Ю
hgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape
Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:
*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/Reshape_1
П
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mulMulhgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency>classification_layers/dense0/batch_normalization/batchnorm/mul*
T0*'
_output_shapes
:џџџџџџџџџ

ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/SumSumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mulegradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ
*
T0
Є
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mul_1Mul*classification_layers/dense0/dense/BiasAddhgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ
*
T0
м
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Sum_1SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/mul_1ggradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0

`gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ю
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes
:
*
T0

Sgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:


Ugradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
щ
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/SumSumjgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1cgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:
*
T0
э
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Sum_1Sumjgradients/classification_layers/dense0/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1egradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
а
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/NegNegSgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
П
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1ReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/NegUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Shape_1*
T0*
_output_shapes
:
*
Tshape0

^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes
:
*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape*
T0
П
hgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/group_deps*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/Reshape_1*
_output_shapes
:
*
T0

Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
Ё
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ы
egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ShapeWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Љ
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mulMulhgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1>classification_layers/dense0/batch_normalization/batchnorm/mul*
_output_shapes
:
*
T0
ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/SumSumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mulegradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ReshapeReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape*
_output_shapes
:
*
Tshape0*
T0
Ѕ
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mul_1Mul8classification_layers/dense0/batch_normalization/Squeezehgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1*
_output_shapes
:
*
T0
м
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Sum_1SumUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/mul_1ggradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1ReshapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Sum_1Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0

`gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/ReshapeZ^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1
С
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshapea^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape
Ч
jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityYgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1a^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:


Mgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   
   
Ъ
Ogradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ReshapeReshapehgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyMgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:


gradients/AddNAddNjgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1jgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*l
_classb
`^loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes
:


Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ShapeConst*
valueB:
*
_output_shapes
:*
dtype0

Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:

х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mulMulgradients/AddN;classification_layers/dense0/batch_normalization/gamma/read*
T0*
_output_shapes
:

а
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/SumSumQgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mulcgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape*
T0*
_output_shapes
:
*
Tshape0
б
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mul_1Mul@classification_layers/dense0/batch_normalization/batchnorm/Rsqrtgradients/AddN*
_output_shapes
:
*
T0
ж
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Sum_1SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/mul_1egradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1ReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Sum_1Ugradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0

^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:
*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape*
T0
П
hgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/Reshape_1*
T0
І
Qgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_likeConst*
valueB
*    *
_output_shapes

:
*
dtype0
о
Mgradients/classification_layers/dense0/batch_normalization/Select_grad/SelectSelect8classification_layers/dense0/batch_normalization/ReshapeOgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/ReshapeQgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_like*
_output_shapes

:
*
T0
р
Ogradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1Select8classification_layers/dense0/batch_normalization/ReshapeQgradients/classification_layers/dense0/batch_normalization/Select_grad/zeros_likeOgradients/classification_layers/dense0/batch_normalization/Squeeze_grad/Reshape*
_output_shapes

:
*
T0

Wgradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_depsNoOpN^gradients/classification_layers/dense0/batch_normalization/Select_grad/SelectP^gradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1

_gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependencyIdentityMgradients/classification_layers/dense0/batch_normalization/Select_grad/SelectX^gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_deps*`
_classV
TRloc:@gradients/classification_layers/dense0/batch_normalization/Select_grad/Select*
_output_shapes

:
*
T0
Ѕ
agradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependency_1IdentityOgradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1X^gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/group_deps*b
_classX
VTloc:@gradients/classification_layers/dense0/batch_normalization/Select_grad/Select_1*
_output_shapes

:
*
T0
Е
Ygradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad@classification_layers/dense0/batch_normalization/batchnorm/Rsqrtfgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency*
_output_shapes
:
*
T0

Pgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:

У
Rgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ReshapeReshape_gradients/classification_layers/dense0/batch_normalization/Select_grad/tuple/control_dependencyPgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/Shape*
T0*
_output_shapes
:
*
Tshape0

Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:

Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
х
cgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ShapeUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
и
Qgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/SumSumYgradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradcgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Л
Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ReshapeReshapeQgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/SumSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape*
_output_shapes
:
*
Tshape0*
T0
м
Sgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Sum_1SumYgradients/classification_layers/dense0/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradegradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Н
Wgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1ReshapeSgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Sum_1Ugradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0

^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_depsNoOpV^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/ReshapeX^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1
Й
fgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependencyIdentityUgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape*
_output_shapes
:

Л
hgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependency_1IdentityWgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1_^gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/group_deps*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
І
Ugradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
Ф
Wgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/ReshapeReshapeRgradients/classification_layers/dense0/batch_normalization/ExpandDims_grad/ReshapeUgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:

 
Ogradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ShapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Ь
Qgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ReshapeReshapefgradients/classification_layers/dense0/batch_normalization/batchnorm/add_grad/tuple/control_dependencyOgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/Shape*
T0*
_output_shapes

:
*
Tshape0
Ѓ
Rgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
Ѕ
Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
т
bgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ShapeTgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
д
Pgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/SumSumWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Reshapebgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
М
Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ReshapeReshapePgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/SumRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape*
Tshape0*
_output_shapes

:
*
T0
и
Rgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Sum_1SumWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_grad/Reshapedgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Т
Vgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1ReshapeRgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Sum_1Tgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0

]gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_depsNoOpU^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/ReshapeW^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1
Й
egradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependencyIdentityTgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape^^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_deps*g
_class]
[Yloc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape*
_output_shapes

:
*
T0
П
ggradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependency_1IdentityVgradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1^^gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1*
_output_shapes

:

Ј
Sgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_likeConst*
dtype0*
_output_shapes

:
*
valueB
*    
ц
Ogradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectSelect:classification_layers/dense0/batch_normalization/Reshape_1Qgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/ReshapeSgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_like*
_output_shapes

:
*
T0
ш
Qgradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1Select:classification_layers/dense0/batch_normalization/Reshape_1Sgradients/classification_layers/dense0/batch_normalization/Select_1_grad/zeros_likeQgradients/classification_layers/dense0/batch_normalization/Squeeze_1_grad/Reshape*
T0*
_output_shapes

:


Ygradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_depsNoOpP^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectR^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1
Ї
agradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependencyIdentityOgradients/classification_layers/dense0/batch_normalization/Select_1_grad/SelectZ^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select*
_output_shapes

:

­
cgradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependency_1IdentityQgradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1Z^gradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/group_deps*
T0*
_output_shapes

:
*d
_classZ
XVloc:@gradients/classification_layers/dense0/batch_normalization/Select_1_grad/Select_1

Rgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ShapeConst*
valueB:
*
_output_shapes
:*
dtype0
Щ
Tgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ReshapeReshapeagradients/classification_layers/dense0/batch_normalization/Select_1_grad/tuple/control_dependencyRgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/Shape*
Tshape0*
_output_shapes
:
*
T0
Ј
Wgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/ShapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
Ъ
Ygradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/ReshapeReshapeTgradients/classification_layers/dense0/batch_normalization/ExpandDims_2_grad/ReshapeWgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Shape*
T0*
_output_shapes

:
*
Tshape0
Ї
Vgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ShapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Љ
Xgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
ю
fgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ShapeXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
о
Tgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/SumSumYgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Reshapefgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ш
Xgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/ReshapeReshapeTgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/SumVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape*
Tshape0*
_output_shapes

:
*
T0
т
Vgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Sum_1SumYgradients/classification_layers/dense0/batch_normalization/moments/Squeeze_1_grad/Reshapehgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ж
Tgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/NegNegVgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Sum_1*
_output_shapes
:*
T0
Ь
Zgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1ReshapeTgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/NegXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

Ё
agradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_depsNoOpY^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape[^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1
Щ
igradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependencyIdentityXgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshapeb^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape*
_output_shapes

:

Я
kgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1IdentityZgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1b^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/group_deps*
_output_shapes

:
*m
_classc
a_loc:@gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/Reshape_1*
T0
о
Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeShapeJclassification_layers/dense0/batch_normalization/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:

Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
І
Rgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/addAddQclassification_layers/dense0/batch_normalization/moments/Mean_1/reduction_indicesSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Size*
T0*
_output_shapes
:
Ќ
Rgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/modFloorModRgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/addSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Size*
_output_shapes
:*
T0
 
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/rangeRangeZgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/startSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/SizeZgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/range/delta*

Tidx0*
_output_shapes
:

Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Г
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/FillFillVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_1Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill/value*
_output_shapes
:*
T0
љ
\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitchDynamicStitchTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/rangeRgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/modTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*
N

Xgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ч
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/MaximumMaximum\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitchXgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ
Ж
Wgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordivFloorDivTgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ShapeVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum*
T0*
_output_shapes
:
л
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ReshapeReshapeigradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency\gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
й
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/TileTileVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ReshapeWgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*

Tmultiples0
р
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_2ShapeJclassification_layers/dense0/batch_normalization/moments/SquaredDifference*
T0*
out_type0*
_output_shapes
:
Ї
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_3Const*
valueB"   
   *
_output_shapes
:*
dtype0

Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ч
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ProdProdVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_2Tgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
 
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ы
Ugradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Prod_1ProdVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Shape_3Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1/yConst*
value	B :*
_output_shapes
: *
dtype0
З
Xgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1MaximumUgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Prod_1Zgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1/y*
_output_shapes
: *
T0
Е
Ygradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv_1FloorDivSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/ProdXgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0
ц
Sgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/CastCastYgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
Н
Vgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truedivRealDivSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/TileSgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ


Tgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul/xConstl^gradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 

Rgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mulMulTgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul/xEclassification_layers/dense0/batch_normalization/moments/shifted_mean*
_output_shapes

:
*
T0
Х
Tgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul_1Mulkgradients/classification_layers/dense0/batch_normalization/moments/variance_grad/tuple/control_dependency_1Rgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul*
_output_shapes

:
*
T0
Щ
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
T0*
out_type0*
_output_shapes
:
В
agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   

ogradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shapeagradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ў
`gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/scalarConstW^gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*
valueB
 *   @*
_output_shapes
: *
dtype0
а
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mulMul`gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/scalarVgradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*'
_output_shapes
:џџџџџџџџџ
*
T0
т
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/subSub*classification_layers/dense0/dense/BiasAddEclassification_layers/dense0/batch_normalization/moments/StopGradientW^gradients/classification_layers/dense0/batch_normalization/moments/Mean_1_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ

ж
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1Mul]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

і
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/SumSum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1ogradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ь
agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/ReshapeReshape]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

њ
_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_1Sum_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/mul_1qgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
щ
cgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape_1Reshape_gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Sum_1agradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

ђ
]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/NegNegcgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:

Ж
jgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_depsNoOpb^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape^^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Neg
і
rgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyIdentityagradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshapek^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*t
_classj
hfloc:@gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Reshape
ч
tgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1Identity]gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Negk^gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps*p
_classf
dbloc:@gradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/Neg*
_output_shapes

:
*
T0
№
gradients/AddN_1AddNegradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependencyTgradients/classification_layers/dense0/batch_normalization/moments/Square_grad/mul_1*g
_class]
[Yloc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape*
_output_shapes

:
*
T0*
N
ж
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ShapeShape<classification_layers/dense0/batch_normalization/moments/Sub*
_output_shapes
:*
out_type0*
T0

Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
И
Xgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/addAddWclassification_layers/dense0/batch_normalization/moments/shifted_mean/reduction_indicesYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size*
T0*
_output_shapes
:
О
Xgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/modFloorModXgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/addYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size*
T0*
_output_shapes
:
І
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Њ
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/rangeRange`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/startYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Size`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/range/delta*
_output_shapes
:*

Tidx0
Ё
_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Х
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/FillFill\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_1_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill/value*
_output_shapes
:*
T0

bgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitchDynamicStitchZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/rangeXgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/modZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ShapeYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*
N
 
^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
й
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/MaximumMaximumbgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitch^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ
Ш
]gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordivFloorDivZgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum*
T0*
_output_shapes
:

\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ReshapeReshapegradients/AddN_1bgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
ы
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/TileTile\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Reshape]gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
и
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_2Shape<classification_layers/dense0/batch_normalization/moments/Sub*
T0*
out_type0*
_output_shapes
:
­
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"   
   
Є
Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
й
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/ProdProd\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_2Zgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
І
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
н
[gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod_1Prod\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Shape_3\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ђ
`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Щ
^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1Maximum[gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod_1`gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1/y*
_output_shapes
: *
T0
Ч
_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv_1FloorDivYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Prod^gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Maximum_1*
_output_shapes
: *
T0
ђ
Ygradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/CastCast_gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
Я
\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivRealDivYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/TileYgradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

Л
Qgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ShapeShape*classification_layers/dense0/dense/BiasAdd*
T0*
_output_shapes
:*
out_type0
Є
Sgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   
п
agradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ShapeSgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
з
Ogradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/SumSum\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivagradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Т
Sgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ReshapeReshapeOgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/SumQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ
*
T0
л
Qgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Sum_1Sum\gradients/classification_layers/dense0/batch_normalization/moments/shifted_mean_grad/truedivcgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ь
Ogradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/NegNegQgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Sum_1*
T0*
_output_shapes
:
Н
Ugradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1ReshapeOgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/NegSgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0

\gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_depsNoOpT^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/ReshapeV^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1
О
dgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependencyIdentitySgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape]^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*f
_class\
ZXloc:@gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape
Л
fgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency_1IdentityUgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1]^gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/Reshape_1*
_output_shapes

:

ќ
gradients/AddN_2AddNggradients/classification_layers/dense0/batch_normalization/moments/mean_grad/tuple/control_dependency_1tgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1fgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency_1*
T0*i
_class_
][loc:@gradients/classification_layers/dense0/batch_normalization/moments/mean_grad/Reshape_1*
N*
_output_shapes

:


gradients/AddN_3AddNhgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyrgradients/classification_layers/dense0/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencydgradients/classification_layers/dense0/batch_normalization/moments/Sub_grad/tuple/control_dependency*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
*
T0*
N
Ђ
Egradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes
:
*
data_formatNHWC*
T0
­
Jgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3F^gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad
л
Rgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3K^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ
*j
_class`
^\loc:@gradients/classification_layers/dense0/batch_normalization/batchnorm/mul_1_grad/Reshape
ѓ
Tgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency_1IdentityEgradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGradK^gradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:
*X
_classN
LJloc:@gradients/classification_layers/dense0/dense/BiasAdd_grad/BiasAddGrad
І
?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulMatMulRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency.classification_layers/dense0/dense/kernel/read*
transpose_b(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
T0

Agradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1MatMulFlatten/ReshapeRgradients/classification_layers/dense0/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	
*
transpose_a(
з
Igradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_depsNoOp@^gradients/classification_layers/dense0/dense/MatMul_grad/MatMulB^gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1
ё
Qgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependencyIdentity?gradients/classification_layers/dense0/dense/MatMul_grad/MatMulJ^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџ*R
_classH
FDloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul
ю
Sgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1IdentityAgradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1J^gradients/classification_layers/dense0/dense/MatMul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/classification_layers/dense0/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0

beta1_power/initial_valueConst*
valueB
 *fff?*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
_output_shapes
: 
­
beta1_power
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
shared_name *
_output_shapes
: *
shape: 
Ь
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel

beta1_power/readIdentitybeta1_power*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
T0

beta2_power/initial_valueConst*
valueB
 *wО?*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
_output_shapes
: 
­
beta2_power
VariableV2*
_output_shapes
: *
dtype0*
shape: *
	container *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
shared_name 
Ь
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
validate_shape(*
_output_shapes
: 

beta2_power/readIdentitybeta2_power*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
T0
е
@classification_layers/dense0/dense/kernel/Adam/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB	
*    *
_output_shapes
:	
*
dtype0
т
.classification_layers/dense0/dense/kernel/Adam
VariableV2*
	container *
dtype0*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
shared_name *
_output_shapes
:	
*
shape:	

Т
5classification_layers/dense0/dense/kernel/Adam/AssignAssign.classification_layers/dense0/dense/kernel/Adam@classification_layers/dense0/dense/kernel/Adam/Initializer/zeros*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	
*
T0*
validate_shape(*
use_locking(
з
3classification_layers/dense0/dense/kernel/Adam/readIdentity.classification_layers/dense0/dense/kernel/Adam*
T0*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel
з
Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
valueB	
*    *
_output_shapes
:	
*
dtype0
ф
0classification_layers/dense0/dense/kernel/Adam_1
VariableV2*
shape:	
*
_output_shapes
:	
*
shared_name *<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
dtype0*
	container 
Ш
7classification_layers/dense0/dense/kernel/Adam_1/AssignAssign0classification_layers/dense0/dense/kernel/Adam_1Bclassification_layers/dense0/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	
*
validate_shape(*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0*
use_locking(
л
5classification_layers/dense0/dense/kernel/Adam_1/readIdentity0classification_layers/dense0/dense/kernel/Adam_1*
_output_shapes
:	
*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0
Ч
>classification_layers/dense0/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    
д
,classification_layers/dense0/dense/bias/Adam
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
Е
3classification_layers/dense0/dense/bias/Adam/AssignAssign,classification_layers/dense0/dense/bias/Adam>classification_layers/dense0/dense/bias/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias
Ь
1classification_layers/dense0/dense/bias/Adam/readIdentity,classification_layers/dense0/dense/bias/Adam*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
_output_shapes
:

Щ
@classification_layers/dense0/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:
*
dtype0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
valueB
*    
ж
.classification_layers/dense0/dense/bias/Adam_1
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *:
_class0
.,loc:@classification_layers/dense0/dense/bias*
shared_name 
Л
5classification_layers/dense0/dense/bias/Adam_1/AssignAssign.classification_layers/dense0/dense/bias/Adam_1@classification_layers/dense0/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@classification_layers/dense0/dense/bias*
validate_shape(*
_output_shapes
:

а
3classification_layers/dense0/dense/bias/Adam_1/readIdentity.classification_layers/dense0/dense/bias/Adam_1*
T0*
_output_shapes
:
*:
_class0
.,loc:@classification_layers/dense0/dense/bias
у
Lclassification_layers/dense0/batch_normalization/beta/Adam/Initializer/zerosConst*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    *
dtype0*
_output_shapes
:

№
:classification_layers/dense0/batch_normalization/beta/Adam
VariableV2*
	container *
shared_name *
dtype0*
shape:
*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta
э
Aclassification_layers/dense0/batch_normalization/beta/Adam/AssignAssign:classification_layers/dense0/batch_normalization/beta/AdamLclassification_layers/dense0/batch_normalization/beta/Adam/Initializer/zeros*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
і
?classification_layers/dense0/batch_normalization/beta/Adam/readIdentity:classification_layers/dense0/batch_normalization/beta/Adam*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
T0
х
Nclassification_layers/dense0/batch_normalization/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
valueB
*    
ђ
<classification_layers/dense0/batch_normalization/beta/Adam_1
VariableV2*
_output_shapes
:
*
dtype0*
shape:
*
	container *H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
shared_name 
ѓ
Cclassification_layers/dense0/batch_normalization/beta/Adam_1/AssignAssign<classification_layers/dense0/batch_normalization/beta/Adam_1Nclassification_layers/dense0/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta
њ
Aclassification_layers/dense0/batch_normalization/beta/Adam_1/readIdentity<classification_layers/dense0/batch_normalization/beta/Adam_1*H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
_output_shapes
:
*
T0
х
Mclassification_layers/dense0/batch_normalization/gamma/Adam/Initializer/zerosConst*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*    *
_output_shapes
:
*
dtype0
ђ
;classification_layers/dense0/batch_normalization/gamma/Adam
VariableV2*
	container *
dtype0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
shared_name *
_output_shapes
:
*
shape:

ё
Bclassification_layers/dense0/batch_normalization/gamma/Adam/AssignAssign;classification_layers/dense0/batch_normalization/gamma/AdamMclassification_layers/dense0/batch_normalization/gamma/Adam/Initializer/zeros*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
љ
@classification_layers/dense0/batch_normalization/gamma/Adam/readIdentity;classification_layers/dense0/batch_normalization/gamma/Adam*
_output_shapes
:
*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
T0
ч
Oclassification_layers/dense0/batch_normalization/gamma/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
valueB
*    *
_output_shapes
:
*
dtype0
є
=classification_layers/dense0/batch_normalization/gamma/Adam_1
VariableV2*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
ї
Dclassification_layers/dense0/batch_normalization/gamma/Adam_1/AssignAssign=classification_layers/dense0/batch_normalization/gamma/Adam_1Oclassification_layers/dense0/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
validate_shape(*
_output_shapes
:

§
Bclassification_layers/dense0/batch_normalization/gamma/Adam_1/readIdentity=classification_layers/dense0/batch_normalization/gamma/Adam_1*
T0*I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
_output_shapes
:

л
Dclassification_layers/dense_last/dense/kernel/Adam/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
_output_shapes

:
*
dtype0
ш
2classification_layers/dense_last/dense/kernel/Adam
VariableV2*
	container *
dtype0*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
shared_name *
_output_shapes

:
*
shape
:

б
9classification_layers/dense_last/dense/kernel/Adam/AssignAssign2classification_layers/dense_last/dense/kernel/AdamDclassification_layers/dense_last/dense/kernel/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel
т
7classification_layers/dense_last/dense/kernel/Adam/readIdentity2classification_layers/dense_last/dense/kernel/Adam*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0
н
Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zerosConst*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
valueB
*    *
dtype0*
_output_shapes

:

ъ
4classification_layers/dense_last/dense/kernel/Adam_1
VariableV2*
shared_name *
shape
:
*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
dtype0*
	container 
з
;classification_layers/dense_last/dense/kernel/Adam_1/AssignAssign4classification_layers/dense_last/dense/kernel/Adam_1Fclassification_layers/dense_last/dense/kernel/Adam_1/Initializer/zeros*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
ц
9classification_layers/dense_last/dense/kernel/Adam_1/readIdentity4classification_layers/dense_last/dense/kernel/Adam_1*
_output_shapes

:
*@
_class6
42loc:@classification_layers/dense_last/dense/kernel*
T0
Я
Bclassification_layers/dense_last/dense/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
_output_shapes
:*
dtype0
м
0classification_layers/dense_last/dense/bias/Adam
VariableV2*
shared_name *>
_class4
20loc:@classification_layers/dense_last/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Х
7classification_layers/dense_last/dense/bias/Adam/AssignAssign0classification_layers/dense_last/dense/bias/AdamBclassification_layers/dense_last/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
validate_shape(*
_output_shapes
:
и
5classification_layers/dense_last/dense/bias/Adam/readIdentity0classification_layers/dense_last/dense/bias/Adam*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0
б
Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
valueB*    *
dtype0*
_output_shapes
:
о
2classification_layers/dense_last/dense/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias
Ы
9classification_layers/dense_last/dense/bias/Adam_1/AssignAssign2classification_layers/dense_last/dense/bias/Adam_1Dclassification_layers/dense_last/dense/bias/Adam_1/Initializer/zeros*>
_class4
20loc:@classification_layers/dense_last/dense/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
м
7classification_layers/dense_last/dense/bias/Adam_1/readIdentity2classification_layers/dense_last/dense/bias/Adam_1*
T0*
_output_shapes
:*>
_class4
20loc:@classification_layers/dense_last/dense/bias
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o:
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
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
_output_shapes
: *
dtype0

?Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam	ApplyAdam)classification_layers/dense0/dense/kernel.classification_layers/dense0/dense/kernel/Adam0classification_layers/dense0/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonSgradients/classification_layers/dense0/dense/MatMul_grad/tuple/control_dependency_1*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
:	
*
T0*
use_nesterov( *
use_locking( 

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

х
KAdam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdam	ApplyAdam5classification_layers/dense0/batch_normalization/beta:classification_layers/dense0/batch_normalization/beta/Adam<classification_layers/dense0/batch_normalization/beta/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonfgradients/classification_layers/dense0/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:
*
use_nesterov( *H
_class>
<:loc:@classification_layers/dense0/batch_normalization/beta*
T0*
use_locking( 
ь
LAdam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdam	ApplyAdam6classification_layers/dense0/batch_normalization/gamma;classification_layers/dense0/batch_normalization/gamma/Adam=classification_layers/dense0/batch_normalization/gamma/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonhgradients/classification_layers/dense0/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_nesterov( *I
_class?
=;loc:@classification_layers/dense0/batch_normalization/gamma*
T0*
use_locking( 
В
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
Ѕ
AAdam/update_classification_layers/dense_last/dense/bias/ApplyAdam	ApplyAdam+classification_layers/dense_last/dense/bias0classification_layers/dense_last/dense/bias/Adam2classification_layers/dense_last/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonXgradients/classification_layers/dense_last/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *>
_class4
20loc:@classification_layers/dense_last/dense/bias*
T0*
use_locking( 
Е
Adam/mulMulbeta1_power/read
Adam/beta1@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*
T0*
_output_shapes
: *<
_class2
0.loc:@classification_layers/dense0/dense/kernel
Д
Adam/AssignAssignbeta1_powerAdam/mul*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
З

Adam/mul_1Mulbeta2_power/read
Adam/beta2@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
_output_shapes
: *
T0
И
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
validate_shape(*<
_class2
0.loc:@classification_layers/dense0/dense/kernel*
T0*
use_locking( 
г
AdamNoOp@^Adam/update_classification_layers/dense0/dense/kernel/ApplyAdam>^Adam/update_classification_layers/dense0/dense/bias/ApplyAdamL^Adam/update_classification_layers/dense0/batch_normalization/beta/ApplyAdamM^Adam/update_classification_layers/dense0/batch_normalization/gamma/ApplyAdamD^Adam/update_classification_layers/dense_last/dense/kernel/ApplyAdamB^Adam/update_classification_layers/dense_last/dense/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
ћ	
initNoOp1^classification_layers/dense0/dense/kernel/Assign/^classification_layers/dense0/dense/bias/Assign=^classification_layers/dense0/batch_normalization/beta/Assign>^classification_layers/dense0/batch_normalization/gamma/AssignD^classification_layers/dense0/batch_normalization/moving_mean/AssignH^classification_layers/dense0/batch_normalization/moving_variance/Assign5^classification_layers/dense_last/dense/kernel/Assign3^classification_layers/dense_last/dense/bias/Assign^beta1_power/Assign^beta2_power/Assign6^classification_layers/dense0/dense/kernel/Adam/Assign8^classification_layers/dense0/dense/kernel/Adam_1/Assign4^classification_layers/dense0/dense/bias/Adam/Assign6^classification_layers/dense0/dense/bias/Adam_1/AssignB^classification_layers/dense0/batch_normalization/beta/Adam/AssignD^classification_layers/dense0/batch_normalization/beta/Adam_1/AssignC^classification_layers/dense0/batch_normalization/gamma/Adam/AssignE^classification_layers/dense0/batch_normalization/gamma/Adam_1/Assign:^classification_layers/dense_last/dense/kernel/Adam/Assign<^classification_layers/dense_last/dense/kernel/Adam_1/Assign8^classification_layers/dense_last/dense/bias/Adam/Assign:^classification_layers/dense_last/dense/bias/Adam_1/Assign""ю
	variablesрн

+classification_layers/dense0/dense/kernel:00classification_layers/dense0/dense/kernel/Assign0classification_layers/dense0/dense/kernel/read:0

)classification_layers/dense0/dense/bias:0.classification_layers/dense0/dense/bias/Assign.classification_layers/dense0/dense/bias/read:0
Е
7classification_layers/dense0/batch_normalization/beta:0<classification_layers/dense0/batch_normalization/beta/Assign<classification_layers/dense0/batch_normalization/beta/read:0
И
8classification_layers/dense0/batch_normalization/gamma:0=classification_layers/dense0/batch_normalization/gamma/Assign=classification_layers/dense0/batch_normalization/gamma/read:0
Ъ
>classification_layers/dense0/batch_normalization/moving_mean:0Cclassification_layers/dense0/batch_normalization/moving_mean/AssignCclassification_layers/dense0/batch_normalization/moving_mean/read:0
ж
Bclassification_layers/dense0/batch_normalization/moving_variance:0Gclassification_layers/dense0/batch_normalization/moving_variance/AssignGclassification_layers/dense0/batch_normalization/moving_variance/read:0

/classification_layers/dense_last/dense/kernel:04classification_layers/dense_last/dense/kernel/Assign4classification_layers/dense_last/dense/kernel/read:0

-classification_layers/dense_last/dense/bias:02classification_layers/dense_last/dense/bias/Assign2classification_layers/dense_last/dense/bias/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
 
0classification_layers/dense0/dense/kernel/Adam:05classification_layers/dense0/dense/kernel/Adam/Assign5classification_layers/dense0/dense/kernel/Adam/read:0
І
2classification_layers/dense0/dense/kernel/Adam_1:07classification_layers/dense0/dense/kernel/Adam_1/Assign7classification_layers/dense0/dense/kernel/Adam_1/read:0

.classification_layers/dense0/dense/bias/Adam:03classification_layers/dense0/dense/bias/Adam/Assign3classification_layers/dense0/dense/bias/Adam/read:0
 
0classification_layers/dense0/dense/bias/Adam_1:05classification_layers/dense0/dense/bias/Adam_1/Assign5classification_layers/dense0/dense/bias/Adam_1/read:0
Ф
<classification_layers/dense0/batch_normalization/beta/Adam:0Aclassification_layers/dense0/batch_normalization/beta/Adam/AssignAclassification_layers/dense0/batch_normalization/beta/Adam/read:0
Ъ
>classification_layers/dense0/batch_normalization/beta/Adam_1:0Cclassification_layers/dense0/batch_normalization/beta/Adam_1/AssignCclassification_layers/dense0/batch_normalization/beta/Adam_1/read:0
Ч
=classification_layers/dense0/batch_normalization/gamma/Adam:0Bclassification_layers/dense0/batch_normalization/gamma/Adam/AssignBclassification_layers/dense0/batch_normalization/gamma/Adam/read:0
Э
?classification_layers/dense0/batch_normalization/gamma/Adam_1:0Dclassification_layers/dense0/batch_normalization/gamma/Adam_1/AssignDclassification_layers/dense0/batch_normalization/gamma/Adam_1/read:0
Ќ
4classification_layers/dense_last/dense/kernel/Adam:09classification_layers/dense_last/dense/kernel/Adam/Assign9classification_layers/dense_last/dense/kernel/Adam/read:0
В
6classification_layers/dense_last/dense/kernel/Adam_1:0;classification_layers/dense_last/dense/kernel/Adam_1/Assign;classification_layers/dense_last/dense/kernel/Adam_1/read:0
І
2classification_layers/dense_last/dense/bias/Adam:07classification_layers/dense_last/dense/bias/Adam/Assign7classification_layers/dense_last/dense/bias/Adam/read:0
Ќ
4classification_layers/dense_last/dense/bias/Adam_1:09classification_layers/dense_last/dense/bias/Adam_1/Assign9classification_layers/dense_last/dense/bias/Adam_1/read:0"g
	summariesZ
X
Evaluation_layers/accuracy:0
Evaluation_layers/loss:0
Evaluation_layers/accuracy_1:0"ъ
trainable_variablesвЯ

+classification_layers/dense0/dense/kernel:00classification_layers/dense0/dense/kernel/Assign0classification_layers/dense0/dense/kernel/read:0

)classification_layers/dense0/dense/bias:0.classification_layers/dense0/dense/bias/Assign.classification_layers/dense0/dense/bias/read:0
Е
7classification_layers/dense0/batch_normalization/beta:0<classification_layers/dense0/batch_normalization/beta/Assign<classification_layers/dense0/batch_normalization/beta/read:0
И
8classification_layers/dense0/batch_normalization/gamma:0=classification_layers/dense0/batch_normalization/gamma/Assign=classification_layers/dense0/batch_normalization/gamma/read:0

/classification_layers/dense_last/dense/kernel:04classification_layers/dense_last/dense/kernel/Assign4classification_layers/dense_last/dense/kernel/read:0

-classification_layers/dense_last/dense/bias:02classification_layers/dense_last/dense/bias/Assign2classification_layers/dense_last/dense/bias/read:0"
train_op

Adam"

update_ops

Bclassification_layers/dense0/batch_normalization/AssignMovingAvg:0
Dclassification_layers/dense0/batch_normalization/AssignMovingAvg_1:0НNr       %:	SЫќЃ^жA*g
!
Evaluation_layers/accuracy(2ќ>

Evaluation_layers/loss_ёч@
#
Evaluation_layers/accuracy_1(2ќ>ыЯЧ@t       _gsв	П0йќЃ^жA*g
!
Evaluation_layers/accuracyЖ1?

Evaluation_layers/losss@@
#
Evaluation_layers/accuracy_1Ж1?ЛEтt       _gsв	цќЃ^жA*g
!
Evaluation_layers/accuracy1?

Evaluation_layers/lossG@
#
Evaluation_layers/accuracy_11?88ыїt       _gsв	Ы8єќЃ^жA*g
!
Evaluation_layers/accuracy?2?

Evaluation_layers/lossEЎX@
#
Evaluation_layers/accuracy_1?2?Ыlъt       _gsв	ћ §Ѓ^жA*g
!
Evaluation_layers/accuracyћ1?

Evaluation_layers/lossr25@
#
Evaluation_layers/accuracy_1ћ1?ЏфЄt       _gsв	ѕЊ§Ѓ^жA*g
!
Evaluation_layers/accuracymn2?

Evaluation_layers/lossЌ.@
#
Evaluation_layers/accuracy_1mn2?MИ&At       _gsв	ПФ§Ѓ^жA*g
!
Evaluation_layers/accuracyЎђ2?

Evaluation_layers/loss28!@
#
Evaluation_layers/accuracy_1Ўђ2?^Bt       _gsв	Ъ"§Ѓ^жA*g
!
Evaluation_layers/accuracy2?

Evaluation_layers/lossЁ!@
#
Evaluation_layers/accuracy_12?>кr|t       _gsв	5-§Ѓ^жA*g
!
Evaluation_layers/accuracyЄо2?

Evaluation_layers/lossдR@
#
Evaluation_layers/accuracy_1Єо2?д(Ўt       _gsв	ѓЄ7§Ѓ^жA	*g
!
Evaluation_layers/accuracy{2?

Evaluation_layers/loss3 @
#
Evaluation_layers/accuracy_1{2? Pt       _gsв	&bJ§Ѓ^жA
*g
!
Evaluation_layers/accuracyЈц2?

Evaluation_layers/losss>л?
#
Evaluation_layers/accuracy_1Јц2?єat       _gsв	1СT§Ѓ^жA*g
!
Evaluation_layers/accuracyЯ4?

Evaluation_layers/lossEжШ?
#
Evaluation_layers/accuracy_1Я4?б9t       _gsв	wH_§Ѓ^жA*g
!
Evaluation_layers/accuracy06?

Evaluation_layers/lossЮнБ?
#
Evaluation_layers/accuracy_106?ЏчбJt       _gsв	7i§Ѓ^жA*g
!
Evaluation_layers/accuracyй;?

Evaluation_layers/loss%Є?
#
Evaluation_layers/accuracy_1й;? q;_t       _gsв	Vt§Ѓ^жA*g
!
Evaluation_layers/accuracyQe8?

Evaluation_layers/lossBv?
#
Evaluation_layers/accuracy_1Qe8?Рпt       _gsв	KБ~§Ѓ^жA*g
!
Evaluation_layers/accuracyЯX7?

Evaluation_layers/lossМБy?
#
Evaluation_layers/accuracy_1ЯX7?АЃt       _gsв	ЃT§Ѓ^жA*g
!
Evaluation_layers/accuracyв/?

Evaluation_layers/loss?
#
Evaluation_layers/accuracy_1в/?PВ?t       _gsв	1ц§Ѓ^жA*g
!
Evaluation_layers/accuracyD&?

Evaluation_layers/lossЎЬ?
#
Evaluation_layers/accuracy_1D&?AяЌt       _gsв	ї§Ѓ^жA*g
!
Evaluation_layers/accuracyЯд&?

Evaluation_layers/loss]aщ?
#
Evaluation_layers/accuracy_1Яд&?st       _gsв	ТЉ§Ѓ^жA*g
!
Evaluation_layers/accuracybС?

Evaluation_layers/loss@
#
Evaluation_layers/accuracy_1bС?Ыt       _gsв	'вМ§Ѓ^жA*g
!
Evaluation_layers/accuracyЌj"?

Evaluation_layers/lossх@
#
Evaluation_layers/accuracy_1Ќj"?mчVt       _gsв	ЏCЧ§Ѓ^жA*g
!
Evaluation_layers/accuracyЊw$?

Evaluation_layers/loss>@
#
Evaluation_layers/accuracy_1Њw$?ЕЋPt       _gsв	ўеб§Ѓ^жA*g
!
Evaluation_layers/accuracyКЪ*?

Evaluation_layers/lossуп?
#
Evaluation_layers/accuracy_1КЪ*?ЖA?t       _gsв	Kєм§Ѓ^жA*g
!
Evaluation_layers/accuracy,?

Evaluation_layers/lossЦФ?
#
Evaluation_layers/accuracy_1,?:аЄt       _gsв	лш§Ѓ^жA*g
!
Evaluation_layers/accuracyЗѕ0?

Evaluation_layers/lossCЛЉ?
#
Evaluation_layers/accuracy_1Зѕ0?нБЌWt       _gsв	Жіѓ§Ѓ^жA*g
!
Evaluation_layers/accuracyъЁ9?

Evaluation_layers/loss6?
#
Evaluation_layers/accuracy_1ъЁ9?jЬшt       _gsв	-;џ§Ѓ^жA*g
!
Evaluation_layers/accuracyЛ3?

Evaluation_layers/loss6з?
#
Evaluation_layers/accuracy_1Л3?n№ цt       _gsв	nЁўЃ^жA*g
!
Evaluation_layers/accuracyЖF;?

Evaluation_layers/loss(?
#
Evaluation_layers/accuracy_1ЖF;?EZЉшt       _gsв	ИћўЃ^жA*g
!
Evaluation_layers/accuracy>?

Evaluation_layers/lossЃР?
#
Evaluation_layers/accuracy_1>?иЅ~t       _gsв	ь'ўЃ^жA*g
!
Evaluation_layers/accuracyBx>?

Evaluation_layers/lossБ^?
#
Evaluation_layers/accuracy_1Bx>?МЇOt       _gsв	ьЅ?ўЃ^жA*g
!
Evaluation_layers/accuracyёЯ=?

Evaluation_layers/lossЩ?
#
Evaluation_layers/accuracy_1ёЯ=?NсХJt       _gsв	WлLўЃ^жA*g
!
Evaluation_layers/accuracy9d>?

Evaluation_layers/loss/:?
#
Evaluation_layers/accuracy_19d>?-t       _gsв	)ZўЃ^жA *g
!
Evaluation_layers/accuracy@t>?

Evaluation_layers/loss:F?
#
Evaluation_layers/accuracy_1@t>?_A"t       _gsв	vgўЃ^жA!*g
!
Evaluation_layers/accuracyГ`??

Evaluation_layers/lossВ?
#
Evaluation_layers/accuracy_1Г`??ЇG3Чt       _gsв	auўЃ^жA"*g
!
Evaluation_layers/accuracyeР>?

Evaluation_layers/losscѕ?
#
Evaluation_layers/accuracy_1eР>?2dГt       _gsв	#ўЃ^жA#*g
!
Evaluation_layers/accuracysм>?

Evaluation_layers/loss3n?
#
Evaluation_layers/accuracy_1sм>?*яot       _gsв	оўЃ^жA$*g
!
Evaluation_layers/accuracyJ>?

Evaluation_layers/lossЧБ?
#
Evaluation_layers/accuracy_1J>?Lt       _gsв	Ї)ўЃ^жA%*g
!
Evaluation_layers/accuracy]А>?

Evaluation_layers/lossЉt?
#
Evaluation_layers/accuracy_1]А>?Дйкt       _gsв	бОЊўЃ^жA&*g
!
Evaluation_layers/accuracyёЯ=?

Evaluation_layers/loss­Є?
#
Evaluation_layers/accuracy_1ёЯ=?рdt       _gsв	1UИўЃ^жA'*g
!
Evaluation_layers/accuracy >?

Evaluation_layers/loss1?
#
Evaluation_layers/accuracy_1 >?щсЖt       _gsв	ЯўЃ^жA(*g
!
Evaluation_layers/accuracyL>?

Evaluation_layers/lossф?
#
Evaluation_layers/accuracy_1L>?ъЮ9ќt       _gsв	ЮтмўЃ^жA)*g
!
Evaluation_layers/accuracy>?

Evaluation_layers/lossј@?
#
Evaluation_layers/accuracy_1>?7ќGt       _gsв	4ъўЃ^жA**g
!
Evaluation_layers/accuracy9d>?

Evaluation_layers/lossјЇ?
#
Evaluation_layers/accuracy_19d>?Cњ§јt       _gsв	ВDїўЃ^жA+*g
!
Evaluation_layers/accuracy №=?

Evaluation_layers/loss!Ђ?
#
Evaluation_layers/accuracy_1 №=?ЌiЏt       _gsв	:јџЃ^жA,*g
!
Evaluation_layers/accuracycМ>?

Evaluation_layers/lossч?
#
Evaluation_layers/accuracy_1cМ>?dКЈt       _gsв	бџЃ^жA-*g
!
Evaluation_layers/accuracyZЈ>?

Evaluation_layers/loss OЂ?
#
Evaluation_layers/accuracy_1ZЈ>?М`t       _gsв	|џЃ^жA.*g
!
Evaluation_layers/accuracy)D>?

Evaluation_layers/loss|Ј?
#
Evaluation_layers/accuracy_1)D>?I1bGt       _gsв	|@#џЃ^жA/*g
!
Evaluation_layers/accuracywф>?

Evaluation_layers/loss/ю?
#
Evaluation_layers/accuracy_1wф>?bZ§t       _gsв	ЕС.џЃ^жA0*g
!
Evaluation_layers/accuracy|№>?

Evaluation_layers/losskл?
#
Evaluation_layers/accuracy_1|№>?ЁІ t       _gsв	Ќ{9џЃ^жA1*g
!
Evaluation_layers/accuracyD|>?

Evaluation_layers/lossГCЄ?
#
Evaluation_layers/accuracy_1D|>?Ј&t       _gsв	MџЃ^жA2*g
!
Evaluation_layers/accuracy-L>?

Evaluation_layers/lossєЋ?
#
Evaluation_layers/accuracy_1-L>?Ht       _gsв	сWџЃ^жA3*g
!
Evaluation_layers/accuracy3X>?

Evaluation_layers/lossИЎЅ?
#
Evaluation_layers/accuracy_13X>?Џ}t       _gsв	UvbџЃ^жA4*g
!
Evaluation_layers/accuracyuр>?

Evaluation_layers/loss^Ё?
#
Evaluation_layers/accuracy_1uр>?Х t       _gsв	ЖmџЃ^жA5*g
!
Evaluation_layers/accuracyј>?

Evaluation_layers/loss!Ј?
#
Evaluation_layers/accuracy_1ј>?^Qфt       _gsв	ЮvwџЃ^жA6*g
!
Evaluation_layers/accuracyј>?

Evaluation_layers/loss-А?
#
Evaluation_layers/accuracy_1ј>?ЏlE@t       _gsв	5#џЃ^жA7*g
!
Evaluation_layers/accuracyqи>?

Evaluation_layers/lossеyИ?
#
Evaluation_layers/accuracy_1qи>?еH?иt       _gsв	РџЃ^жA8*g
!
Evaluation_layers/accuracyJ>?

Evaluation_layers/lossЃЛИ?
#
Evaluation_layers/accuracy_1J>?H\пТt       _gsв	fZџЃ^жA9*g
!
Evaluation_layers/accuracygФ>?

Evaluation_layers/loss$=Г?
#
Evaluation_layers/accuracy_1gФ>?Г7t       _gsв	:ЦЁџЃ^жA:*g
!
Evaluation_layers/accuracysм>?

Evaluation_layers/loss(Р?
#
Evaluation_layers/accuracy_1sм>?Цxt       _gsв	tЌџЃ^жA;*g
!
Evaluation_layers/accuracyXЄ>?

Evaluation_layers/loss"Г?
#
Evaluation_layers/accuracy_1XЄ>?fTqt       _gsв	МПџЃ^жA<*g
!
Evaluation_layers/accuracyL>?

Evaluation_layers/loss(Д?
#
Evaluation_layers/accuracy_1L>?
t       _gsв	ЂѕЩџЃ^жA=*g
!
Evaluation_layers/accuracy<l>?

Evaluation_layers/lossЛ?
#
Evaluation_layers/accuracy_1<l>?ЭћЃt       _gsв	0ЭдџЃ^жA>*g
!
Evaluation_layers/accuracy|№>?

Evaluation_layers/lossЭЉР?
#
Evaluation_layers/accuracy_1|№>?§Lbћt       _gsв	э^пџЃ^жA?*g
!
Evaluation_layers/accuracyј>?

Evaluation_layers/loss]Х?
#
Evaluation_layers/accuracy_1ј>?LЕt       _gsв	VЈщџЃ^жA@*g
!
Evaluation_layers/accuracy-L>?

Evaluation_layers/lossБЁХ?
#
Evaluation_layers/accuracy_1-L>?8R6t       _gsв	ЕєџЃ^жAA*g
!
Evaluation_layers/accuracyZЈ>?

Evaluation_layers/loss?ќЪ?
#
Evaluation_layers/accuracy_1ZЈ>? Сt       _gsв	uЬџџЃ^жAB*g
!
Evaluation_layers/accuracyeР>?

Evaluation_layers/loss0$б?
#
Evaluation_layers/accuracy_1eР>?*t       _gsв	г Є^жAC*g
!
Evaluation_layers/accuracycМ>?

Evaluation_layers/lossOд?
#
Evaluation_layers/accuracy_1cМ>?AџWЏt       _gsв	y Є^жAD*g
!
Evaluation_layers/accuracy>p>?

Evaluation_layers/lossД^ж?
#
Evaluation_layers/accuracy_1>p>?яУщt       _gsв	' Є^жAE*g
!
Evaluation_layers/accuracyF>?

Evaluation_layers/lossн(н?
#
Evaluation_layers/accuracy_1F>?5иЃВt       _gsв	+№> Є^жAF*g
!
Evaluation_layers/accuracyP>?

Evaluation_layers/lossрЋе?
#
Evaluation_layers/accuracy_1P>?;pчt       _gsв	RТL Є^жAG*g
!
Evaluation_layers/accuracy(>?

Evaluation_layers/loss8Vт?
#
Evaluation_layers/accuracy_1(>?Й!E|