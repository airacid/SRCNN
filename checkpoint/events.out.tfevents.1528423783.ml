       ЃK"	  РYyЦжAbrain.Event:2ЏЇбyc      9Тrо	я№ФYyЦжA"ьЦ
y
imagesPlaceholder*/
_output_shapes
:џџџџџџџџџ!!*$
shape:џџџџџџџџџ!!*
dtype0
y
labelsPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
І
,conv1/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv1/kernel*%
valueB"	   	      @   *
dtype0*
_output_shapes
:

+conv1/kernel/Initializer/random_normal/meanConst*
_class
loc:@conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-conv1/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv1/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv1/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:		@*

seed *
T0*
_class
loc:@conv1/kernel*
seed2 
я
*conv1/kernel/Initializer/random_normal/mulMul;conv1/kernel/Initializer/random_normal/RandomStandardNormal-conv1/kernel/Initializer/random_normal/stddev*&
_output_shapes
:		@*
T0*
_class
loc:@conv1/kernel
и
&conv1/kernel/Initializer/random_normalAdd*conv1/kernel/Initializer/random_normal/mul+conv1/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
Б
conv1/kernel
VariableV2*
dtype0*&
_output_shapes
:		@*
shared_name *
_class
loc:@conv1/kernel*
	container *
shape:		@
Ю
conv1/kernel/AssignAssignconv1/kernel&conv1/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(*&
_output_shapes
:		@
}
conv1/kernel/readIdentityconv1/kernel*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@

conv1/bias/Initializer/zerosConst*
_class
loc:@conv1/bias*
valueB@*    *
dtype0*
_output_shapes
:@


conv1/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv1/bias
В
conv1/bias/AssignAssign
conv1/biasconv1/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1/bias*
validate_shape(
k
conv1/bias/readIdentity
conv1/bias*
T0*
_class
loc:@conv1/bias*
_output_shapes
:@
d
conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
к
conv1/Conv2DConv2Dimagesconv1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

conv1/BiasAddBiasAddconv1/Conv2Dconv1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ@*
T0
[

conv1/ReluReluconv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ@
І
,conv2/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv2/kernel*%
valueB"      @       *
dtype0*
_output_shapes
:

+conv2/kernel/Initializer/random_normal/meanConst*
_class
loc:@conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-conv2/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv2/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv2/kernel/Initializer/random_normal/shape*

seed *
T0*
_class
loc:@conv2/kernel*
seed2 *
dtype0*&
_output_shapes
:@ 
я
*conv2/kernel/Initializer/random_normal/mulMul;conv2/kernel/Initializer/random_normal/RandomStandardNormal-conv2/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
и
&conv2/kernel/Initializer/random_normalAdd*conv2/kernel/Initializer/random_normal/mul+conv2/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
Б
conv2/kernel
VariableV2*
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@conv2/kernel*
	container 
Ю
conv2/kernel/AssignAssignconv2/kernel&conv2/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@conv2/kernel*
validate_shape(*&
_output_shapes
:@ 
}
conv2/kernel/readIdentityconv2/kernel*&
_output_shapes
:@ *
T0*
_class
loc:@conv2/kernel

conv2/bias/Initializer/zerosConst*
_output_shapes
: *
_class
loc:@conv2/bias*
valueB *    *
dtype0


conv2/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv2/bias*
	container 
В
conv2/bias/AssignAssign
conv2/biasconv2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: 
k
conv2/bias/readIdentity
conv2/bias*
T0*
_class
loc:@conv2/bias*
_output_shapes
: 
d
conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
о
conv2/Conv2DConv2D
conv1/Reluconv2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ 

conv2/BiasAddBiasAddconv2/Conv2Dconv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
[

conv2/ReluReluconv2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ 
Є
+pred/kernel/Initializer/random_normal/shapeConst*
_class
loc:@pred/kernel*%
valueB"             *
dtype0*
_output_shapes
:

*pred/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
_class
loc:@pred/kernel*
valueB
 *    *
dtype0

,pred/kernel/Initializer/random_normal/stddevConst*
_class
loc:@pred/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
і
:pred/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+pred/kernel/Initializer/random_normal/shape*
T0*
_class
loc:@pred/kernel*
seed2 *
dtype0*&
_output_shapes
: *

seed 
ы
)pred/kernel/Initializer/random_normal/mulMul:pred/kernel/Initializer/random_normal/RandomStandardNormal,pred/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
д
%pred/kernel/Initializer/random_normalAdd)pred/kernel/Initializer/random_normal/mul*pred/kernel/Initializer/random_normal/mean*
_class
loc:@pred/kernel*&
_output_shapes
: *
T0
Џ
pred/kernel
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@pred/kernel*
	container 
Ъ
pred/kernel/AssignAssignpred/kernel%pred/kernel/Initializer/random_normal*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
z
pred/kernel/readIdentitypred/kernel*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 

pred/bias/Initializer/zerosConst*
_class
loc:@pred/bias*
valueB*    *
dtype0*
_output_shapes
:

	pred/bias
VariableV2*
shared_name *
_class
loc:@pred/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ў
pred/bias/AssignAssign	pred/biaspred/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pred/bias*
validate_shape(*
_output_shapes
:
h
pred/bias/readIdentity	pred/bias*
T0*
_class
loc:@pred/bias*
_output_shapes
:
c
pred/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
м
pred/Conv2DConv2D
conv2/Relupred/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations


pred/BiasAddBiasAddpred/Conv2Dpred/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
Z
subSublabelspred/BiasAdd*/
_output_shapes
:џџџџџџџџџ*
T0
O
SquareSquaresub*
T0*/
_output_shapes
:џџџџџџџџџ
^
ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Y
MeanMeanSquareConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
`
global_step_1/tagsConst*
valueB Bglobal_step_1*
dtype0*
_output_shapes
: 
e
global_step_1ScalarSummaryglobal_step_1/tagsglobal_step/read*
_output_shapes
: *
T0	
c
ExponentialDecay/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
_
ExponentialDecay/CastCastglobal_step/read*
_output_shapes
: *

DstT0*

SrcT0	
]
ExponentialDecay/Cast_1/xConst*
valueB	 : *
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
^
ExponentialDecay/Cast_2/xConst*
_output_shapes
: *
valueB
 *Тu?*
dtype0
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
z
!gradients/Mean_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
_
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
Є
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*/
_output_shapes
:џџџџџџџџџ
a
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*/
_output_shapes
:џџџџџџџџџ*
T0
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
|
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*/
_output_shapes
:џџџџџџџџџ*
T0

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*/
_output_shapes
:џџџџџџџџџ*
T0
^
gradients/sub_grad/ShapeShapelabels*
T0*
out_type0*
_output_shapes
:
f
gradients/sub_grad/Shape_1Shapepred/BiasAdd*
T0*
out_type0*
_output_shapes
:
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
Ј
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ѓ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
т
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
ш
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*/
_output_shapes
:џџџџџџџџџ
Ё
'gradients/pred/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0

,gradients/pred/BiasAdd_grad/tuple/group_depsNoOp(^gradients/pred/BiasAdd_grad/BiasAddGrad.^gradients/sub_grad/tuple/control_dependency_1

4gradients/pred/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/sub_grad/tuple/control_dependency_1-^gradients/pred/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
ћ
6gradients/pred/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/pred/BiasAdd_grad/BiasAddGrad-^gradients/pred/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*:
_class0
.,loc:@gradients/pred/BiasAdd_grad/BiasAddGrad

!gradients/pred/Conv2D_grad/ShapeNShapeN
conv2/Relupred/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
y
 gradients/pred/Conv2D_grad/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0
є
.gradients/pred/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/pred/Conv2D_grad/ShapeNpred/kernel/read4gradients/pred/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Ы
/gradients/pred/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv2/Relu gradients/pred/Conv2D_grad/Const4gradients/pred/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
: 

+gradients/pred/Conv2D_grad/tuple/group_depsNoOp0^gradients/pred/Conv2D_grad/Conv2DBackpropFilter/^gradients/pred/Conv2D_grad/Conv2DBackpropInput

3gradients/pred/Conv2D_grad/tuple/control_dependencyIdentity.gradients/pred/Conv2D_grad/Conv2DBackpropInput,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/pred/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 

5gradients/pred/Conv2D_grad/tuple/control_dependency_1Identity/gradients/pred/Conv2D_grad/Conv2DBackpropFilter,^gradients/pred/Conv2D_grad/tuple/group_deps*B
_class8
64loc:@gradients/pred/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
Љ
"gradients/conv2/Relu_grad/ReluGradReluGrad3gradients/pred/Conv2D_grad/tuple/control_dependency
conv2/Relu*/
_output_shapes
:џџџџџџџџџ *
T0

(gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

-gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv2/BiasAdd_grad/BiasAddGrad#^gradients/conv2/Relu_grad/ReluGrad

5gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv2/Relu_grad/ReluGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ *
T0
џ
7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv2/BiasAdd_grad/BiasAddGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

"gradients/conv2/Conv2D_grad/ShapeNShapeN
conv1/Reluconv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv2/Conv2D_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
ј
/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv2/Conv2D_grad/ShapeNconv2/kernel/read5gradients/conv2/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
Ю
0gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv1/Relu!gradients/conv2/Conv2D_grad/Const5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

,gradients/conv2/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2/Conv2D_grad/Conv2DBackpropInput

4gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv2/Conv2D_grad/Conv2DBackpropInput-^gradients/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ@*
T0*B
_class8
64loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropInput

6gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv2/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
Њ
"gradients/conv1/Relu_grad/ReluGradReluGrad4gradients/conv2/Conv2D_grad/tuple/control_dependency
conv1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ@

(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@

-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv1/BiasAdd_grad/BiasAddGrad#^gradients/conv1/Relu_grad/ReluGrad

5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv1/Relu_grad/ReluGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ@*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad
џ
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad

"gradients/conv1/Conv2D_grad/ShapeNShapeNimagesconv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv1/Conv2D_grad/ConstConst*%
valueB"	   	      @   *
dtype0*
_output_shapes
:
ј
/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/kernel/read5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Ъ
0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterimages!gradients/conv1/Conv2D_grad/Const5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:		@*
	dilations


,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ!!

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:		@
b
GradientDescent/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 

8GradientDescent/update_conv1/kernel/ApplyGradientDescentApplyGradientDescentconv1/kernelGradientDescent/learning_rate6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@

6GradientDescent/update_conv1/bias/ApplyGradientDescentApplyGradientDescent
conv1/biasGradientDescent/learning_rate7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv1/bias*
_output_shapes
:@

8GradientDescent/update_conv2/kernel/ApplyGradientDescentApplyGradientDescentconv2/kernelGradientDescent/learning_rate6gradients/conv2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:@ *
use_locking( *
T0*
_class
loc:@conv2/kernel

6GradientDescent/update_conv2/bias/ApplyGradientDescentApplyGradientDescent
conv2/biasGradientDescent/learning_rate7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2/bias*
_output_shapes
: *
use_locking( *
T0

7GradientDescent/update_pred/kernel/ApplyGradientDescentApplyGradientDescentpred/kernelGradientDescent/learning_rate5gradients/pred/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: *
use_locking( *
T0*
_class
loc:@pred/kernel

5GradientDescent/update_pred/bias/ApplyGradientDescentApplyGradientDescent	pred/biasGradientDescent/learning_rate6gradients/pred/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@pred/bias*
_output_shapes
:*
use_locking( 
ј
GradientDescent/updateNoOp7^GradientDescent/update_conv1/bias/ApplyGradientDescent9^GradientDescent/update_conv1/kernel/ApplyGradientDescent7^GradientDescent/update_conv2/bias/ApplyGradientDescent9^GradientDescent/update_conv2/kernel/ApplyGradientDescent6^GradientDescent/update_pred/bias/ApplyGradientDescent8^GradientDescent/update_pred/kernel/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
_class
loc:@global_step*
value	B	 R*
dtype0	*
_output_shapes
: 

GradientDescent	AssignAddglobal_stepGradientDescent/value*
T0	*
_class
loc:@global_step*
_output_shapes
: *
use_locking( "МпУw      Wi>я	>IХYyЦжAJЖя
Ї
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
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
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
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
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.8.02v1.8.0-0-g93bc2e2072ьЦ
y
imagesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ!!*$
shape:џџџџџџџџџ!!
y
labelsPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
І
,conv1/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv1/kernel*%
valueB"	   	      @   *
dtype0*
_output_shapes
:

+conv1/kernel/Initializer/random_normal/meanConst*
_class
loc:@conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-conv1/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv1/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv1/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:		@*

seed *
T0*
_class
loc:@conv1/kernel*
seed2 
я
*conv1/kernel/Initializer/random_normal/mulMul;conv1/kernel/Initializer/random_normal/RandomStandardNormal-conv1/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
и
&conv1/kernel/Initializer/random_normalAdd*conv1/kernel/Initializer/random_normal/mul+conv1/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
Б
conv1/kernel
VariableV2*
dtype0*&
_output_shapes
:		@*
shared_name *
_class
loc:@conv1/kernel*
	container *
shape:		@
Ю
conv1/kernel/AssignAssignconv1/kernel&conv1/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(*&
_output_shapes
:		@
}
conv1/kernel/readIdentityconv1/kernel*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@

conv1/bias/Initializer/zerosConst*
_class
loc:@conv1/bias*
valueB@*    *
dtype0*
_output_shapes
:@


conv1/bias
VariableV2*
shared_name *
_class
loc:@conv1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
В
conv1/bias/AssignAssign
conv1/biasconv1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/bias*
validate_shape(*
_output_shapes
:@
k
conv1/bias/readIdentity
conv1/bias*
_output_shapes
:@*
T0*
_class
loc:@conv1/bias
d
conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
к
conv1/Conv2DConv2Dimagesconv1/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@*
	dilations


conv1/BiasAddBiasAddconv1/Conv2Dconv1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ@
[

conv1/ReluReluconv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ@
І
,conv2/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv2/kernel*%
valueB"      @       *
dtype0*
_output_shapes
:

+conv2/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
_class
loc:@conv2/kernel*
valueB
 *    *
dtype0

-conv2/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv2/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv2/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:@ *

seed *
T0*
_class
loc:@conv2/kernel*
seed2 
я
*conv2/kernel/Initializer/random_normal/mulMul;conv2/kernel/Initializer/random_normal/RandomStandardNormal-conv2/kernel/Initializer/random_normal/stddev*&
_output_shapes
:@ *
T0*
_class
loc:@conv2/kernel
и
&conv2/kernel/Initializer/random_normalAdd*conv2/kernel/Initializer/random_normal/mul+conv2/kernel/Initializer/random_normal/mean*&
_output_shapes
:@ *
T0*
_class
loc:@conv2/kernel
Б
conv2/kernel
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@conv2/kernel*
	container *
shape:@ 
Ю
conv2/kernel/AssignAssignconv2/kernel&conv2/kernel/Initializer/random_normal*
T0*
_class
loc:@conv2/kernel*
validate_shape(*&
_output_shapes
:@ *
use_locking(
}
conv2/kernel/readIdentityconv2/kernel*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 

conv2/bias/Initializer/zerosConst*
_class
loc:@conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 


conv2/bias
VariableV2*
shared_name *
_class
loc:@conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
В
conv2/bias/AssignAssign
conv2/biasconv2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: 
k
conv2/bias/readIdentity
conv2/bias*
_output_shapes
: *
T0*
_class
loc:@conv2/bias
d
conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
о
conv2/Conv2DConv2D
conv1/Reluconv2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ 

conv2/BiasAddBiasAddconv2/Conv2Dconv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
[

conv2/ReluReluconv2/BiasAdd*/
_output_shapes
:џџџџџџџџџ *
T0
Є
+pred/kernel/Initializer/random_normal/shapeConst*
_class
loc:@pred/kernel*%
valueB"             *
dtype0*
_output_shapes
:

*pred/kernel/Initializer/random_normal/meanConst*
_class
loc:@pred/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,pred/kernel/Initializer/random_normal/stddevConst*
_class
loc:@pred/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
і
:pred/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+pred/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
: *

seed *
T0*
_class
loc:@pred/kernel*
seed2 
ы
)pred/kernel/Initializer/random_normal/mulMul:pred/kernel/Initializer/random_normal/RandomStandardNormal,pred/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
д
%pred/kernel/Initializer/random_normalAdd)pred/kernel/Initializer/random_normal/mul*pred/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
Џ
pred/kernel
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@pred/kernel*
	container 
Ъ
pred/kernel/AssignAssignpred/kernel%pred/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: 
z
pred/kernel/readIdentitypred/kernel*&
_output_shapes
: *
T0*
_class
loc:@pred/kernel

pred/bias/Initializer/zerosConst*
_class
loc:@pred/bias*
valueB*    *
dtype0*
_output_shapes
:

	pred/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pred/bias*
	container *
shape:
Ў
pred/bias/AssignAssign	pred/biaspred/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pred/bias*
validate_shape(*
_output_shapes
:
h
pred/bias/readIdentity	pred/bias*
_output_shapes
:*
T0*
_class
loc:@pred/bias
c
pred/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
м
pred/Conv2DConv2D
conv2/Relupred/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

pred/BiasAddBiasAddpred/Conv2Dpred/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
Z
subSublabelspred/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
O
SquareSquaresub*
T0*/
_output_shapes
:џџџџџџџџџ
^
ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Y
MeanMeanSquareConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
`
global_step_1/tagsConst*
valueB Bglobal_step_1*
dtype0*
_output_shapes
: 
e
global_step_1ScalarSummaryglobal_step_1/tagsglobal_step/read*
T0	*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
_output_shapes
: *
valueB
 *Зб8*
dtype0
_
ExponentialDecay/CastCastglobal_step/read*
_output_shapes
: *

DstT0*

SrcT0	
]
ExponentialDecay/Cast_1/xConst*
valueB	 : *
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
^
ExponentialDecay/Cast_2/xConst*
valueB
 *Тu?*
dtype0*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
z
!gradients/Mean_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*&
_output_shapes
:*
T0*
Tshape0
_
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
Є
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*/
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
a
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*/
_output_shapes
:џџџџџџџџџ
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
|
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0*/
_output_shapes
:џџџџџџџџџ

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*/
_output_shapes
:џџџџџџџџџ
^
gradients/sub_grad/ShapeShapelabels*
_output_shapes
:*
T0*
out_type0
f
gradients/sub_grad/Shape_1Shapepred/BiasAdd*
T0*
out_type0*
_output_shapes
:
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Є
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ј
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ѓ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
т
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
ш
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
Ё
'gradients/pred/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0

,gradients/pred/BiasAdd_grad/tuple/group_depsNoOp(^gradients/pred/BiasAdd_grad/BiasAddGrad.^gradients/sub_grad/tuple/control_dependency_1

4gradients/pred/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/sub_grad/tuple/control_dependency_1-^gradients/pred/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*/
_output_shapes
:џџџџџџџџџ
ћ
6gradients/pred/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/pred/BiasAdd_grad/BiasAddGrad-^gradients/pred/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pred/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

!gradients/pred/Conv2D_grad/ShapeNShapeN
conv2/Relupred/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
y
 gradients/pred/Conv2D_grad/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0
є
.gradients/pred/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/pred/Conv2D_grad/ShapeNpred/kernel/read4gradients/pred/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides

Ы
/gradients/pred/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv2/Relu gradients/pred/Conv2D_grad/Const4gradients/pred/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

+gradients/pred/Conv2D_grad/tuple/group_depsNoOp0^gradients/pred/Conv2D_grad/Conv2DBackpropFilter/^gradients/pred/Conv2D_grad/Conv2DBackpropInput

3gradients/pred/Conv2D_grad/tuple/control_dependencyIdentity.gradients/pred/Conv2D_grad/Conv2DBackpropInput,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/pred/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 

5gradients/pred/Conv2D_grad/tuple/control_dependency_1Identity/gradients/pred/Conv2D_grad/Conv2DBackpropFilter,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pred/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Љ
"gradients/conv2/Relu_grad/ReluGradReluGrad3gradients/pred/Conv2D_grad/tuple/control_dependency
conv2/Relu*
T0*/
_output_shapes
:џџџџџџџџџ 

(gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

-gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv2/BiasAdd_grad/BiasAddGrad#^gradients/conv2/Relu_grad/ReluGrad

5gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv2/Relu_grad/ReluGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ 
џ
7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv2/BiasAdd_grad/BiasAddGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

"gradients/conv2/Conv2D_grad/ShapeNShapeN
conv1/Reluconv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv2/Conv2D_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
ј
/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv2/Conv2D_grad/ShapeNconv2/kernel/read5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ю
0gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv1/Relu!gradients/conv2/Conv2D_grad/Const5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

,gradients/conv2/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2/Conv2D_grad/Conv2DBackpropInput

4gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv2/Conv2D_grad/Conv2DBackpropInput-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ@

6gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv2/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
Њ
"gradients/conv1/Relu_grad/ReluGradReluGrad4gradients/conv2/Conv2D_grad/tuple/control_dependency
conv1/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0

(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@

-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv1/BiasAdd_grad/BiasAddGrad#^gradients/conv1/Relu_grad/ReluGrad

5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv1/Relu_grad/ReluGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ@
џ
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

"gradients/conv1/Conv2D_grad/ShapeNShapeNimagesconv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv1/Conv2D_grad/ConstConst*%
valueB"	   	      @   *
dtype0*
_output_shapes
:
ј
/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/kernel/read5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ъ
0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterimages!gradients/conv1/Conv2D_grad/Const5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*&
_output_shapes
:		@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ!!*
T0

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:		@
b
GradientDescent/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 

8GradientDescent/update_conv1/kernel/ApplyGradientDescentApplyGradientDescentconv1/kernelGradientDescent/learning_rate6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@*
use_locking( 

6GradientDescent/update_conv1/bias/ApplyGradientDescentApplyGradientDescent
conv1/biasGradientDescent/learning_rate7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@conv1/bias

8GradientDescent/update_conv2/kernel/ApplyGradientDescentApplyGradientDescentconv2/kernelGradientDescent/learning_rate6gradients/conv2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:@ *
use_locking( *
T0*
_class
loc:@conv2/kernel

6GradientDescent/update_conv2/bias/ApplyGradientDescentApplyGradientDescent
conv2/biasGradientDescent/learning_rate7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv2/bias*
_output_shapes
: 

7GradientDescent/update_pred/kernel/ApplyGradientDescentApplyGradientDescentpred/kernelGradientDescent/learning_rate5gradients/pred/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: *
use_locking( *
T0*
_class
loc:@pred/kernel

5GradientDescent/update_pred/bias/ApplyGradientDescentApplyGradientDescent	pred/biasGradientDescent/learning_rate6gradients/pred/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pred/bias*
_output_shapes
:
ј
GradientDescent/updateNoOp7^GradientDescent/update_conv1/bias/ApplyGradientDescent9^GradientDescent/update_conv1/kernel/ApplyGradientDescent7^GradientDescent/update_conv2/bias/ApplyGradientDescent9^GradientDescent/update_conv2/kernel/ApplyGradientDescent6^GradientDescent/update_pred/bias/ApplyGradientDescent8^GradientDescent/update_pred/kernel/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R

GradientDescent	AssignAddglobal_stepGradientDescent/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: ""Ч
trainable_variablesЏЌ
d
conv1/kernel:0conv1/kernel/Assignconv1/kernel/read:02(conv1/kernel/Initializer/random_normal:0
T
conv1/bias:0conv1/bias/Assignconv1/bias/read:02conv1/bias/Initializer/zeros:0
d
conv2/kernel:0conv2/kernel/Assignconv2/kernel/read:02(conv2/kernel/Initializer/random_normal:0
T
conv2/bias:0conv2/bias/Assignconv2/bias/read:02conv2/bias/Initializer/zeros:0
`
pred/kernel:0pred/kernel/Assignpred/kernel/read:02'pred/kernel/Initializer/random_normal:0
P
pred/bias:0pred/bias/Assignpred/bias/read:02pred/bias/Initializer/zeros:0"(
	summaries

loss:0
global_step_1:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"
train_op

GradientDescent"
	variables
d
conv1/kernel:0conv1/kernel/Assignconv1/kernel/read:02(conv1/kernel/Initializer/random_normal:0
T
conv1/bias:0conv1/bias/Assignconv1/bias/read:02conv1/bias/Initializer/zeros:0
d
conv2/kernel:0conv2/kernel/Assignconv2/kernel/read:02(conv2/kernel/Initializer/random_normal:0
T
conv2/bias:0conv2/bias/Assignconv2/bias/read:02conv2/bias/Initializer/zeros:0
`
pred/kernel:0pred/kernel/Assignpred/kernel/read:02'pred/kernel/Initializer/random_normal:0
P
pred/bias:0pred/bias/Assignpred/bias/read:02pred/bias/Initializer/zeros:0
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0ш+:5Й      u%Ч	hrѓYyЦжA"џё
y
imagesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ!!*$
shape:џџџџџџџџџ!!
y
labelsPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
І
,conv1/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv1/kernel*%
valueB"	   	      @   *
dtype0*
_output_shapes
:

+conv1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
_class
loc:@conv1/kernel*
valueB
 *    

-conv1/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv1/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv1/kernel/Initializer/random_normal/shape*
T0*
_class
loc:@conv1/kernel*
seed2 *
dtype0*&
_output_shapes
:		@*

seed 
я
*conv1/kernel/Initializer/random_normal/mulMul;conv1/kernel/Initializer/random_normal/RandomStandardNormal-conv1/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
и
&conv1/kernel/Initializer/random_normalAdd*conv1/kernel/Initializer/random_normal/mul+conv1/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
Б
conv1/kernel
VariableV2*
shared_name *
_class
loc:@conv1/kernel*
	container *
shape:		@*
dtype0*&
_output_shapes
:		@
Ю
conv1/kernel/AssignAssignconv1/kernel&conv1/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(*&
_output_shapes
:		@
}
conv1/kernel/readIdentityconv1/kernel*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@

conv1/bias/Initializer/zerosConst*
_output_shapes
:@*
_class
loc:@conv1/bias*
valueB@*    *
dtype0


conv1/bias
VariableV2*
shared_name *
_class
loc:@conv1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
В
conv1/bias/AssignAssign
conv1/biasconv1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv1/bias*
validate_shape(*
_output_shapes
:@
k
conv1/bias/readIdentity
conv1/bias*
T0*
_class
loc:@conv1/bias*
_output_shapes
:@
d
conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
к
conv1/Conv2DConv2Dimagesconv1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@

conv1/BiasAddBiasAddconv1/Conv2Dconv1/bias/read*/
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC
[

conv1/ReluReluconv1/BiasAdd*/
_output_shapes
:џџџџџџџџџ@*
T0
І
,conv2/kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
_class
loc:@conv2/kernel*%
valueB"      @       *
dtype0

+conv2/kernel/Initializer/random_normal/meanConst*
_class
loc:@conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-conv2/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv2/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv2/kernel/Initializer/random_normal/shape*
T0*
_class
loc:@conv2/kernel*
seed2 *
dtype0*&
_output_shapes
:@ *

seed 
я
*conv2/kernel/Initializer/random_normal/mulMul;conv2/kernel/Initializer/random_normal/RandomStandardNormal-conv2/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
и
&conv2/kernel/Initializer/random_normalAdd*conv2/kernel/Initializer/random_normal/mul+conv2/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
Б
conv2/kernel
VariableV2*
_class
loc:@conv2/kernel*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name 
Ю
conv2/kernel/AssignAssignconv2/kernel&conv2/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@conv2/kernel*
validate_shape(*&
_output_shapes
:@ 
}
conv2/kernel/readIdentityconv2/kernel*
_class
loc:@conv2/kernel*&
_output_shapes
:@ *
T0

conv2/bias/Initializer/zerosConst*
_class
loc:@conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 


conv2/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@conv2/bias
В
conv2/bias/AssignAssign
conv2/biasconv2/bias/Initializer/zeros*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
k
conv2/bias/readIdentity
conv2/bias*
_output_shapes
: *
T0*
_class
loc:@conv2/bias
d
conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
о
conv2/Conv2DConv2D
conv1/Reluconv2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ 

conv2/BiasAddBiasAddconv2/Conv2Dconv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
[

conv2/ReluReluconv2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ 
Є
+pred/kernel/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@pred/kernel*%
valueB"             

*pred/kernel/Initializer/random_normal/meanConst*
_output_shapes
: *
_class
loc:@pred/kernel*
valueB
 *    *
dtype0

,pred/kernel/Initializer/random_normal/stddevConst*
_output_shapes
: *
_class
loc:@pred/kernel*
valueB
 *o:*
dtype0
і
:pred/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+pred/kernel/Initializer/random_normal/shape*
seed2 *
dtype0*&
_output_shapes
: *

seed *
T0*
_class
loc:@pred/kernel
ы
)pred/kernel/Initializer/random_normal/mulMul:pred/kernel/Initializer/random_normal/RandomStandardNormal,pred/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
д
%pred/kernel/Initializer/random_normalAdd)pred/kernel/Initializer/random_normal/mul*pred/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
Џ
pred/kernel
VariableV2*
_class
loc:@pred/kernel*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
Ъ
pred/kernel/AssignAssignpred/kernel%pred/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: 
z
pred/kernel/readIdentitypred/kernel*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 

pred/bias/Initializer/zerosConst*
_class
loc:@pred/bias*
valueB*    *
dtype0*
_output_shapes
:

	pred/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@pred/bias
Ў
pred/bias/AssignAssign	pred/biaspred/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pred/bias*
validate_shape(
h
pred/bias/readIdentity	pred/bias*
T0*
_class
loc:@pred/bias*
_output_shapes
:
c
pred/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
м
pred/Conv2DConv2D
conv2/Relupred/kernel/read*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

pred/BiasAddBiasAddpred/Conv2Dpred/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
Z
subSublabelspred/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
O
SquareSquaresub*/
_output_shapes
:џџџџџџџџџ*
T0
^
ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Y
MeanMeanSquareConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 

global_step/Initializer/zerosConst*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R 

global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
`
global_step_1/tagsConst*
valueB Bglobal_step_1*
dtype0*
_output_shapes
: 
e
global_step_1ScalarSummaryglobal_step_1/tagsglobal_step/read*
T0	*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
_
ExponentialDecay/CastCastglobal_step/read*
_output_shapes
: *

DstT0*

SrcT0	
]
ExponentialDecay/Cast_1/xConst*
valueB	 : *
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
^
ExponentialDecay/Cast_2/xConst*
_output_shapes
: *
valueB
 *Тu?*
dtype0
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
_output_shapes
: *
T0
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
z
!gradients/Mean_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
_
gradients/Mean_grad/ShapeShapeSquare*
_output_shapes
:*
T0*
out_type0
Є
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*/
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
a
gradients/Mean_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*/
_output_shapes
:џџџџџџџџџ
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
|
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*/
_output_shapes
:џџџџџџџџџ*
T0

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*
T0*/
_output_shapes
:џџџџџџџџџ
^
gradients/sub_grad/ShapeShapelabels*
out_type0*
_output_shapes
:*
T0
f
gradients/sub_grad/Shape_1Shapepred/BiasAdd*
out_type0*
_output_shapes
:*
T0
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Є
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
Ј
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ѓ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
т
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
ш
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
Ё
'gradients/pred/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0

,gradients/pred/BiasAdd_grad/tuple/group_depsNoOp(^gradients/pred/BiasAdd_grad/BiasAddGrad.^gradients/sub_grad/tuple/control_dependency_1

4gradients/pred/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/sub_grad/tuple/control_dependency_1-^gradients/pred/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
ћ
6gradients/pred/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/pred/BiasAdd_grad/BiasAddGrad-^gradients/pred/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pred/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0

!gradients/pred/Conv2D_grad/ShapeNShapeN
conv2/Relupred/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
y
 gradients/pred/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
є
.gradients/pred/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/pred/Conv2D_grad/ShapeNpred/kernel/read4gradients/pred/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Ы
/gradients/pred/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv2/Relu gradients/pred/Conv2D_grad/Const4gradients/pred/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

+gradients/pred/Conv2D_grad/tuple/group_depsNoOp0^gradients/pred/Conv2D_grad/Conv2DBackpropFilter/^gradients/pred/Conv2D_grad/Conv2DBackpropInput

3gradients/pred/Conv2D_grad/tuple/control_dependencyIdentity.gradients/pred/Conv2D_grad/Conv2DBackpropInput,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/pred/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 

5gradients/pred/Conv2D_grad/tuple/control_dependency_1Identity/gradients/pred/Conv2D_grad/Conv2DBackpropFilter,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pred/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Љ
"gradients/conv2/Relu_grad/ReluGradReluGrad3gradients/pred/Conv2D_grad/tuple/control_dependency
conv2/Relu*/
_output_shapes
:џџџџџџџџџ *
T0

(gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0

-gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv2/BiasAdd_grad/BiasAddGrad#^gradients/conv2/Relu_grad/ReluGrad

5gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv2/Relu_grad/ReluGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ 
џ
7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv2/BiasAdd_grad/BiasAddGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*;
_class1
/-loc:@gradients/conv2/BiasAdd_grad/BiasAddGrad

"gradients/conv2/Conv2D_grad/ShapeNShapeN
conv1/Reluconv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv2/Conv2D_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
ј
/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv2/Conv2D_grad/ShapeNconv2/kernel/read5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ю
0gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv1/Relu!gradients/conv2/Conv2D_grad/Const5gradients/conv2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

,gradients/conv2/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2/Conv2D_grad/Conv2DBackpropInput

4gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv2/Conv2D_grad/Conv2DBackpropInput-^gradients/conv2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ@*
T0*B
_class8
64loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropInput

6gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv2/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
Њ
"gradients/conv1/Relu_grad/ReluGradReluGrad4gradients/conv2/Conv2D_grad/tuple/control_dependency
conv1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ@

(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@

-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv1/BiasAdd_grad/BiasAddGrad#^gradients/conv1/Relu_grad/ReluGrad

5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv1/Relu_grad/ReluGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ@*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad
џ
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

"gradients/conv1/Conv2D_grad/ShapeNShapeNimagesconv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv1/Conv2D_grad/ConstConst*%
valueB"	   	      @   *
dtype0*
_output_shapes
:
ј
/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/kernel/read5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ
0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterimages!gradients/conv1/Conv2D_grad/Const5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:		@*
	dilations


,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ!!*
T0*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:		@
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *Зб8

8GradientDescent/update_conv1/kernel/ApplyGradientDescentApplyGradientDescentconv1/kernelGradientDescent/learning_rate6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@*
use_locking( 

6GradientDescent/update_conv1/bias/ApplyGradientDescentApplyGradientDescent
conv1/biasGradientDescent/learning_rate7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@conv1/bias

8GradientDescent/update_conv2/kernel/ApplyGradientDescentApplyGradientDescentconv2/kernelGradientDescent/learning_rate6gradients/conv2/Conv2D_grad/tuple/control_dependency_1*
_class
loc:@conv2/kernel*&
_output_shapes
:@ *
use_locking( *
T0

6GradientDescent/update_conv2/bias/ApplyGradientDescentApplyGradientDescent
conv2/biasGradientDescent/learning_rate7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@conv2/bias

7GradientDescent/update_pred/kernel/ApplyGradientDescentApplyGradientDescentpred/kernelGradientDescent/learning_rate5gradients/pred/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 

5GradientDescent/update_pred/bias/ApplyGradientDescentApplyGradientDescent	pred/biasGradientDescent/learning_rate6gradients/pred/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@pred/bias*
_output_shapes
:*
use_locking( 
ј
GradientDescent/updateNoOp7^GradientDescent/update_conv1/bias/ApplyGradientDescent9^GradientDescent/update_conv1/kernel/ApplyGradientDescent7^GradientDescent/update_conv2/bias/ApplyGradientDescent9^GradientDescent/update_conv2/kernel/ApplyGradientDescent6^GradientDescent/update_pred/bias/ApplyGradientDescent8^GradientDescent/update_pred/kernel/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
_class
loc:@global_step*
value	B	 R*
dtype0	*
_output_shapes
: 

GradientDescent	AssignAddglobal_stepGradientDescent/value*
T0	*
_class
loc:@global_step*
_output_shapes
: *
use_locking( 

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
_output_shapes
: *
T0

_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
_output_shapes
: *
T0

h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
T0
*
_output_shapes
: 
b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
T0	*
N*
_output_shapes
: : 
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
T0	*
_output_shapes
: 

initNoOp^conv1/bias/Assign^conv1/kernel/Assign^conv2/bias/Assign^conv2/kernel/Assign^global_step/Assign^pred/bias/Assign^pred/kernel/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
Ё
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedconv1/kernel*
_class
loc:@conv1/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized
conv1/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv1/bias
Ѓ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2/kernel*
dtype0*
_output_shapes
: *
_class
loc:@conv2/kernel

6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized
conv2/bias*
_class
loc:@conv2/bias*
dtype0*
_output_shapes
: 
Ё
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedpred/kernel*
_class
loc:@pred/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized	pred/bias*
_output_shapes
: *
_class
loc:@pred/bias*
dtype0
Ё
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
ћ
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_6"/device:CPU:0*
N*
_output_shapes
:*
T0
*

axis 

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
г
$report_uninitialized_variables/ConstConst"/device:CPU:0*l
valuecBaBconv1/kernelB
conv1/biasBconv2/kernelB
conv2/biasBpred/kernelB	pred/biasBglobal_step*
dtype0*
_output_shapes
:

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
_output_shapes
:*
valueB: *
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
_output_shapes
:*
valueB: *
dtype0

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:*
T0
*
Tshape0
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:џџџџџџџџџ*
T0

Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims


9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0*
N
Ѓ
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedconv1/kernel*
_class
loc:@conv1/kernel*
dtype0*
_output_shapes
: 
Ё
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized
conv1/bias*
_class
loc:@conv1/bias*
dtype0*
_output_shapes
: 
Ѕ
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2/kernel*
_class
loc:@conv2/kernel*
dtype0*
_output_shapes
: 
Ё
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized
conv2/bias*
_class
loc:@conv2/bias*
dtype0*
_output_shapes
: 
Ѓ
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedpred/kernel*
_class
loc:@pred/kernel*
dtype0*
_output_shapes
: 

8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized	pred/bias*
_class
loc:@pred/bias*
dtype0*
_output_shapes
: 
Ѓ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
е
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*l
valuecBaBconv1/kernelB
conv1/biasBconv2/kernelB
conv2/biasBpred/kernelB	pred/biasBglobal_step*
dtype0*
_output_shapes
:

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask 
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ

init_2NoOp

init_all_tablesNoOp

init_3NoOp
8
group_deps_1NoOp^init_2^init_3^init_all_tables
X
Merge/MergeSummaryMergeSummarylossglobal_step_1*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_c795aa43950b416f876b0d9c91fb61bd/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ч
save/SaveV2/tensor_namesConst"/device:CPU:0*l
valuecBaB
conv1/biasBconv1/kernelB
conv2/biasBconv2/kernelBglobal_stepB	pred/biasBpred/kernel*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
т
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
conv1/biasconv1/kernel
conv2/biasconv2/kernelglobal_step	pred/biaspred/kernel"/device:CPU:0*
dtypes
	2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ъ
save/RestoreV2/tensor_namesConst"/device:CPU:0*l
valuecBaB
conv1/biasBconv1/kernelB
conv2/biasBconv2/kernelBglobal_stepB	pred/biasBpred/kernel*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
Н
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	

save/AssignAssign
conv1/biassave/RestoreV2*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1/bias*
validate_shape(
В
save/Assign_1Assignconv1/kernelsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(*&
_output_shapes
:		@
Ђ
save/Assign_2Assign
conv2/biassave/RestoreV2:2*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
В
save/Assign_3Assignconv2/kernelsave/RestoreV2:3*
_class
loc:@conv2/kernel*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0
 
save/Assign_4Assignglobal_stepsave/RestoreV2:4*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
 
save/Assign_5Assign	pred/biassave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@pred/bias*
validate_shape(*
_output_shapes
:
А
save/Assign_6Assignpred/kernelsave/RestoreV2:6*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard"єФBэи      {ўX[	РѓYyЦжAJБ
*ь)
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
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
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype


LogicalNot
x

y

;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
\
	RefSwitch
data"T
pred

output_false"T
output_true"T"	
Ttype
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	
*1.8.02v1.8.0-0-g93bc2e2072џё
y
imagesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ!!*$
shape:џџџџџџџџџ!!
y
labelsPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
І
,conv1/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv1/kernel*%
valueB"	   	      @   *
dtype0*
_output_shapes
:

+conv1/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
_class
loc:@conv1/kernel*
valueB
 *    

-conv1/kernel/Initializer/random_normal/stddevConst*
_class
loc:@conv1/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
љ
;conv1/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv1/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:		@*

seed *
T0*
_class
loc:@conv1/kernel*
seed2 
я
*conv1/kernel/Initializer/random_normal/mulMul;conv1/kernel/Initializer/random_normal/RandomStandardNormal-conv1/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@
и
&conv1/kernel/Initializer/random_normalAdd*conv1/kernel/Initializer/random_normal/mul+conv1/kernel/Initializer/random_normal/mean*
_class
loc:@conv1/kernel*&
_output_shapes
:		@*
T0
Б
conv1/kernel
VariableV2*
dtype0*&
_output_shapes
:		@*
shared_name *
_class
loc:@conv1/kernel*
	container *
shape:		@
Ю
conv1/kernel/AssignAssignconv1/kernel&conv1/kernel/Initializer/random_normal*&
_output_shapes
:		@*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(
}
conv1/kernel/readIdentityconv1/kernel*
_class
loc:@conv1/kernel*&
_output_shapes
:		@*
T0

conv1/bias/Initializer/zerosConst*
_class
loc:@conv1/bias*
valueB@*    *
dtype0*
_output_shapes
:@


conv1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv1/bias*
	container *
shape:@
В
conv1/bias/AssignAssign
conv1/biasconv1/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1/bias*
validate_shape(
k
conv1/bias/readIdentity
conv1/bias*
T0*
_class
loc:@conv1/bias*
_output_shapes
:@
d
conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
к
conv1/Conv2DConv2Dimagesconv1/kernel/read*
paddingVALID*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

conv1/BiasAddBiasAddconv1/Conv2Dconv1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ@
[

conv1/ReluReluconv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ@
І
,conv2/kernel/Initializer/random_normal/shapeConst*
_class
loc:@conv2/kernel*%
valueB"      @       *
dtype0*
_output_shapes
:

+conv2/kernel/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *
_class
loc:@conv2/kernel*
valueB
 *    

-conv2/kernel/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *
_class
loc:@conv2/kernel*
valueB
 *o:
љ
;conv2/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal,conv2/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
:@ *

seed *
T0*
_class
loc:@conv2/kernel*
seed2 
я
*conv2/kernel/Initializer/random_normal/mulMul;conv2/kernel/Initializer/random_normal/RandomStandardNormal-conv2/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
и
&conv2/kernel/Initializer/random_normalAdd*conv2/kernel/Initializer/random_normal/mul+conv2/kernel/Initializer/random_normal/mean*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 
Б
conv2/kernel
VariableV2*
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@conv2/kernel*
	container 
Ю
conv2/kernel/AssignAssignconv2/kernel&conv2/kernel/Initializer/random_normal*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@conv2/kernel
}
conv2/kernel/readIdentityconv2/kernel*
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 

conv2/bias/Initializer/zerosConst*
_class
loc:@conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 


conv2/bias
VariableV2*
shared_name *
_class
loc:@conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
В
conv2/bias/AssignAssign
conv2/biasconv2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: 
k
conv2/bias/readIdentity
conv2/bias*
T0*
_class
loc:@conv2/bias*
_output_shapes
: 
d
conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
о
conv2/Conv2DConv2D
conv1/Reluconv2/kernel/read*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

conv2/BiasAddBiasAddconv2/Conv2Dconv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
[

conv2/ReluReluconv2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ 
Є
+pred/kernel/Initializer/random_normal/shapeConst*
_class
loc:@pred/kernel*%
valueB"             *
dtype0*
_output_shapes
:

*pred/kernel/Initializer/random_normal/meanConst*
_class
loc:@pred/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,pred/kernel/Initializer/random_normal/stddevConst*
_class
loc:@pred/kernel*
valueB
 *o:*
dtype0*
_output_shapes
: 
і
:pred/kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormal+pred/kernel/Initializer/random_normal/shape*
dtype0*&
_output_shapes
: *

seed *
T0*
_class
loc:@pred/kernel*
seed2 
ы
)pred/kernel/Initializer/random_normal/mulMul:pred/kernel/Initializer/random_normal/RandomStandardNormal,pred/kernel/Initializer/random_normal/stddev*
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 
д
%pred/kernel/Initializer/random_normalAdd)pred/kernel/Initializer/random_normal/mul*pred/kernel/Initializer/random_normal/mean*&
_output_shapes
: *
T0*
_class
loc:@pred/kernel
Џ
pred/kernel
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@pred/kernel
Ъ
pred/kernel/AssignAssignpred/kernel%pred/kernel/Initializer/random_normal*
use_locking(*
T0*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: 
z
pred/kernel/readIdentitypred/kernel*&
_output_shapes
: *
T0*
_class
loc:@pred/kernel

pred/bias/Initializer/zerosConst*
_class
loc:@pred/bias*
valueB*    *
dtype0*
_output_shapes
:

	pred/bias
VariableV2*
shared_name *
_class
loc:@pred/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ў
pred/bias/AssignAssign	pred/biaspred/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@pred/bias*
validate_shape(*
_output_shapes
:
h
pred/bias/readIdentity	pred/bias*
T0*
_class
loc:@pred/bias*
_output_shapes
:
c
pred/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
м
pred/Conv2DConv2D
conv2/Relupred/kernel/read*/
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

pred/BiasAddBiasAddpred/Conv2Dpred/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
Z
subSublabelspred/BiasAdd*/
_output_shapes
:џџџџџџџџџ*
T0
O
SquareSquaresub*
T0*/
_output_shapes
:џџџџџџџџџ
^
ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Y
MeanMeanSquareConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
`
global_step_1/tagsConst*
valueB Bglobal_step_1*
dtype0*
_output_shapes
: 
e
global_step_1ScalarSummaryglobal_step_1/tagsglobal_step/read*
_output_shapes
: *
T0	
c
ExponentialDecay/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
_
ExponentialDecay/CastCastglobal_step/read*
_output_shapes
: *

DstT0*

SrcT0	
]
ExponentialDecay/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB	 : 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*
_output_shapes
: *

DstT0
^
ExponentialDecay/Cast_2/xConst*
valueB
 *Тu?*
dtype0*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
z
!gradients/Mean_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*&
_output_shapes
:*
T0*
Tshape0
_
gradients/Mean_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
Є
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*/
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
a
gradients/Mean_grad/Shape_1ShapeSquare*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*/
_output_shapes
:џџџџџџџџџ
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
|
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*/
_output_shapes
:џџџџџџџџџ*
T0

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*/
_output_shapes
:џџџџџџџџџ*
T0
^
gradients/sub_grad/ShapeShapelabels*
_output_shapes
:*
T0*
out_type0
f
gradients/sub_grad/Shape_1Shapepred/BiasAdd*
out_type0*
_output_shapes
:*
T0
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*/
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ј
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
Ѓ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
т
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
ш
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*/
_output_shapes
:џџџџџџџџџ*
T0
Ё
'gradients/pred/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/sub_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:

,gradients/pred/BiasAdd_grad/tuple/group_depsNoOp(^gradients/pred/BiasAdd_grad/BiasAddGrad.^gradients/sub_grad/tuple/control_dependency_1

4gradients/pred/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/sub_grad/tuple/control_dependency_1-^gradients/pred/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*/
_output_shapes
:џџџџџџџџџ
ћ
6gradients/pred/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/pred/BiasAdd_grad/BiasAddGrad-^gradients/pred/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*:
_class0
.,loc:@gradients/pred/BiasAdd_grad/BiasAddGrad

!gradients/pred/Conv2D_grad/ShapeNShapeN
conv2/Relupred/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
y
 gradients/pred/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
є
.gradients/pred/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/pred/Conv2D_grad/ShapeNpred/kernel/read4gradients/pred/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
Ы
/gradients/pred/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv2/Relu gradients/pred/Conv2D_grad/Const4gradients/pred/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

+gradients/pred/Conv2D_grad/tuple/group_depsNoOp0^gradients/pred/Conv2D_grad/Conv2DBackpropFilter/^gradients/pred/Conv2D_grad/Conv2DBackpropInput

3gradients/pred/Conv2D_grad/tuple/control_dependencyIdentity.gradients/pred/Conv2D_grad/Conv2DBackpropInput,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/pred/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 

5gradients/pred/Conv2D_grad/tuple/control_dependency_1Identity/gradients/pred/Conv2D_grad/Conv2DBackpropFilter,^gradients/pred/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pred/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Љ
"gradients/conv2/Relu_grad/ReluGradReluGrad3gradients/pred/Conv2D_grad/tuple/control_dependency
conv2/Relu*
T0*/
_output_shapes
:џџџџџџџџџ 

(gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

-gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv2/BiasAdd_grad/BiasAddGrad#^gradients/conv2/Relu_grad/ReluGrad

5gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv2/Relu_grad/ReluGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *
T0*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad
џ
7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv2/BiasAdd_grad/BiasAddGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

"gradients/conv2/Conv2D_grad/ShapeNShapeN
conv1/Reluconv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv2/Conv2D_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
ј
/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv2/Conv2D_grad/ShapeNconv2/kernel/read5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Ю
0gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
conv1/Relu!gradients/conv2/Conv2D_grad/Const5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*&
_output_shapes
:@ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

,gradients/conv2/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2/Conv2D_grad/Conv2DBackpropInput

4gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv2/Conv2D_grad/Conv2DBackpropInput-^gradients/conv2/Conv2D_grad/tuple/group_deps*B
_class8
64loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ@*
T0

6gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv2/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@ 
Њ
"gradients/conv1/Relu_grad/ReluGradReluGrad4gradients/conv2/Conv2D_grad/tuple/control_dependency
conv1/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0

(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/conv1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0

-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp)^gradients/conv1/BiasAdd_grad/BiasAddGrad#^gradients/conv1/Relu_grad/ReluGrad

5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/conv1/Relu_grad/ReluGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ@
џ
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@

"gradients/conv1/Conv2D_grad/ShapeNShapeNimagesconv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
z
!gradients/conv1/Conv2D_grad/ConstConst*%
valueB"	   	      @   *
dtype0*
_output_shapes
:
ј
/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/kernel/read5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ
0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterimages!gradients/conv1/Conv2D_grad/Const5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:		@

,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ!!

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*&
_output_shapes
:		@*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter
b
GradientDescent/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 

8GradientDescent/update_conv1/kernel/ApplyGradientDescentApplyGradientDescentconv1/kernelGradientDescent/learning_rate6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv1/kernel*&
_output_shapes
:		@

6GradientDescent/update_conv1/bias/ApplyGradientDescentApplyGradientDescent
conv1/biasGradientDescent/learning_rate7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv1/bias*
_output_shapes
:@

8GradientDescent/update_conv2/kernel/ApplyGradientDescentApplyGradientDescentconv2/kernelGradientDescent/learning_rate6gradients/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv2/kernel*&
_output_shapes
:@ 

6GradientDescent/update_conv2/bias/ApplyGradientDescentApplyGradientDescent
conv2/biasGradientDescent/learning_rate7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@conv2/bias*
_output_shapes
: 

7GradientDescent/update_pred/kernel/ApplyGradientDescentApplyGradientDescentpred/kernelGradientDescent/learning_rate5gradients/pred/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@pred/kernel*&
_output_shapes
: 

5GradientDescent/update_pred/bias/ApplyGradientDescentApplyGradientDescent	pred/biasGradientDescent/learning_rate6gradients/pred/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
_class
loc:@pred/bias
ј
GradientDescent/updateNoOp7^GradientDescent/update_conv1/bias/ApplyGradientDescent9^GradientDescent/update_conv1/kernel/ApplyGradientDescent7^GradientDescent/update_conv2/bias/ApplyGradientDescent9^GradientDescent/update_conv2/kernel/ApplyGradientDescent6^GradientDescent/update_pred/bias/ApplyGradientDescent8^GradientDescent/update_pred/kernel/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
_class
loc:@global_step*
value	B	 R*
dtype0	*
_output_shapes
: 

GradientDescent	AssignAddglobal_stepGradientDescent/value*
_output_shapes
: *
use_locking( *
T0	*
_class
loc:@global_step

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
T0
*
_output_shapes
: : 
a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
T0
*
_output_shapes
: 
b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
_output_shapes
: *
T0	

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
N*
_output_shapes
: : *
T0	
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
_output_shapes
: *
T0	

initNoOp^conv1/bias/Assign^conv1/kernel/Assign^conv2/bias/Assign^conv2/kernel/Assign^global_step/Assign^pred/bias/Assign^pred/kernel/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
Ё
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedconv1/kernel*
_class
loc:@conv1/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized
conv1/bias*
_class
loc:@conv1/bias*
dtype0*
_output_shapes
: 
Ѓ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedconv2/kernel*
_class
loc:@conv2/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized
conv2/bias*
dtype0*
_output_shapes
: *
_class
loc:@conv2/bias
Ё
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedpred/kernel*
_output_shapes
: *
_class
loc:@pred/kernel*
dtype0

6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized	pred/bias*
_class
loc:@pred/bias*
dtype0*
_output_shapes
: 
Ё
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
ћ
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_6"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
г
$report_uninitialized_variables/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
:*l
valuecBaBconv1/kernelB
conv1/biasBconv2/kernelB
conv2/biasBpred/kernelB	pred/biasBglobal_step

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB: *
dtype0

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
_output_shapes
:*
valueB:*
dtype0
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
_output_shapes
:*

Tidx0*
T0*
N
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:џџџџџџџџџ*
T0

Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ

9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*
N*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Ѓ
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedconv1/kernel*
_class
loc:@conv1/kernel*
dtype0*
_output_shapes
: 
Ё
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized
conv1/bias*
_class
loc:@conv1/bias*
dtype0*
_output_shapes
: 
Ѕ
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedconv2/kernel*
_output_shapes
: *
_class
loc:@conv2/kernel*
dtype0
Ё
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized
conv2/bias*
_class
loc:@conv2/bias*
dtype0*
_output_shapes
: 
Ѓ
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedpred/kernel*
_class
loc:@pred/kernel*
dtype0*
_output_shapes
: 

8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized	pred/bias*
dtype0*
_output_shapes
: *
_class
loc:@pred/bias
Ѓ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_6"/device:CPU:0*
_output_shapes
:*
T0
*

axis *
N

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
е
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*l
valuecBaBconv1/kernelB
conv1/biasBconv2/kernelB
conv2/biasBpred/kernelB	pred/biasBglobal_step*
dtype0*
_output_shapes
:

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
_output_shapes
:*
T0*

axis *
N

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:*
T0
*
Tshape0
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0

init_2NoOp

init_all_tablesNoOp

init_3NoOp
8
group_deps_1NoOp^init_2^init_3^init_all_tables
X
Merge/MergeSummaryMergeSummarylossglobal_step_1*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_c795aa43950b416f876b0d9c91fb61bd/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ч
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*l
valuecBaB
conv1/biasBconv1/kernelB
conv2/biasBconv2/kernelBglobal_stepB	pred/biasBpred/kernel*
dtype0

save/SaveV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
т
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
conv1/biasconv1/kernel
conv2/biasconv2/kernelglobal_step	pred/biaspred/kernel"/device:CPU:0*
dtypes
	2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ъ
save/RestoreV2/tensor_namesConst"/device:CPU:0*l
valuecBaB
conv1/biasBconv1/kernelB
conv2/biasBconv2/kernelBglobal_stepB	pred/biasBpred/kernel*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*!
valueBB B B B B B B 
Н
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	

save/AssignAssign
conv1/biassave/RestoreV2*
T0*
_class
loc:@conv1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
В
save/Assign_1Assignconv1/kernelsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@conv1/kernel*
validate_shape(*&
_output_shapes
:		@
Ђ
save/Assign_2Assign
conv2/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@conv2/bias*
validate_shape(*
_output_shapes
: 
В
save/Assign_3Assignconv2/kernelsave/RestoreV2:3*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@conv2/kernel*
validate_shape(
 
save/Assign_4Assignglobal_stepsave/RestoreV2:4*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
 
save/Assign_5Assign	pred/biassave/RestoreV2:5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pred/bias
А
save/Assign_6Assignpred/kernelsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@pred/kernel*
validate_shape(*&
_output_shapes
: 

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"&

summary_op

Merge/MergeSummary:0"(
	summaries

loss:0
global_step_1:0"Ч
trainable_variablesЏЌ
d
conv1/kernel:0conv1/kernel/Assignconv1/kernel/read:02(conv1/kernel/Initializer/random_normal:0
T
conv1/bias:0conv1/bias/Assignconv1/bias/read:02conv1/bias/Initializer/zeros:0
d
conv2/kernel:0conv2/kernel/Assignconv2/kernel/read:02(conv2/kernel/Initializer/random_normal:0
T
conv2/bias:0conv2/bias/Assignconv2/bias/read:02conv2/bias/Initializer/zeros:0
`
pred/kernel:0pred/kernel/Assignpred/kernel/read:02'pred/kernel/Initializer/random_normal:0
P
pred/bias:0pred/bias/Assignpred/bias/read:02pred/bias/Initializer/zeros:0"W
ready_for_local_init_op<
:
8report_uninitialized_variables_1/boolean_mask/GatherV2:0"
init_op


group_deps"Р
cond_contextЏЌ

global_step/cond/cond_textglobal_step/cond/pred_id:0global_step/cond/switch_t:0 *Ј
global_step/cond/pred_id:0
global_step/cond/read/Switch:1
global_step/cond/read:0
global_step/cond/switch_t:0
global_step:08
global_step/cond/pred_id:0global_step/cond/pred_id:0:
global_step/cond/switch_t:0global_step/cond/switch_t:0/
global_step:0global_step/cond/read/Switch:1
Є
global_step/cond/cond_text_1global_step/cond/pred_id:0global_step/cond/switch_f:0*Ъ
global_step/Initializer/zeros:0
global_step/cond/Switch_1:0
global_step/cond/Switch_1:1
global_step/cond/pred_id:0
global_step/cond/switch_f:0:
global_step/cond/switch_f:0global_step/cond/switch_f:0>
global_step/Initializer/zeros:0global_step/cond/Switch_1:08
global_step/cond/pred_id:0global_step/cond/pred_id:0"!
local_init_op

group_deps_1"
	variables
d
conv1/kernel:0conv1/kernel/Assignconv1/kernel/read:02(conv1/kernel/Initializer/random_normal:0
T
conv1/bias:0conv1/bias/Assignconv1/bias/read:02conv1/bias/Initializer/zeros:0
d
conv2/kernel:0conv2/kernel/Assignconv2/kernel/read:02(conv2/kernel/Initializer/random_normal:0
T
conv2/bias:0conv2/bias/Assignconv2/bias/read:02conv2/bias/Initializer/zeros:0
`
pred/kernel:0pred/kernel/Assignpred/kernel/read:02'pred/kernel/Initializer/random_normal:0
P
pred/bias:0pred/bias/Assignpred/bias/read:02pred/bias/Initializer/zeros:0
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"
ready_op


concat:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"2
global_step_read_op_cache

global_step/add:0"
train_op

GradientDescent"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:04а*       $d%	ZyЦжAёб:п| !3       Ї V	PZyЦжAёб*#

lossЩР;

global_step_1РЃJ?RjM       =cдІ	s'ZyЦжAёб:=9/media/airacid/Data/project/3_SRCNN/checkpoint/model.ckptEp(       џpJ	{њ+ZyЦжAев*

global_step/secЬ]CЃЄ3       Ї V	,ZyЦжAев*#

losspS:

global_step_1PЅJЃyG(       џpJ	GJZyЦжAЙг*

global_step/secE;SC
]E3       Ї V	сSJZyЦжAЙг*#

loss@g:

global_step_1рІJoб(       џpJ	ѓUkZyЦжAд*

global_step/sec~ACOБ Ј3       Ї V	TckZyЦжAд*#

lossEO;

global_step_1pЈJх№](       џpJ	
hZyЦжAе*

global_step/secЮACјЃ/N3       Ї V	вvZyЦжAе*#

lossєgс;

global_step_1 ЊJF.ЙD