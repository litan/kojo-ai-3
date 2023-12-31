import scala.util.Using
import scala.jdk.CollectionConverters._

import ai.djl.engine.Engine
import ai.djl.ndarray._
import ai.djl.ndarray.types.DataType
import ai.djl.nn.Activation
import ai.djl.pytorch.jni.JniUtils
import ai.djl.training.dataset.Dataset
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.util.ProgressBar

import net.kogics.kojo.preprocess.StandardScaler
import net.kogics.kojo.nn._

Engine.getInstance
JniUtils.setGradMode(false)
JniUtils.setGraphExecutorOptimize(false)
resetGradientCollection()
