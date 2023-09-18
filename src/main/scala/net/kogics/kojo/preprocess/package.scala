/*
 * Copyright (C) 2019 Lalit Pant <pant.lalit@gmail.com>
 *
 * The contents of this file are subject to the GNU General Public License
 * Version 3 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of
 * the License at http://www.gnu.org/copyleft/gpl.html
 *
 * Software distributed under the License is distributed on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * rights and limitations under the License.
 *
 */
package net.kogics.kojo

import org.apache.commons.math3.stat.StatUtils

package object preprocess {
  trait Scaler {
    def fit(data: Array[Double]): Unit
    def transform(data: Array[Double]): Array[Double]
    def inverseTransform(data: Array[Double]): Array[Double]
    def fitTransform(data: Array[Double]): Array[Double] = {
      fit(data)
      transform(data)
    }
  }

  class StandardScaler extends Scaler {
    var mean = 0.0
    var std = 0.0
    def fit(data: Array[Double]) = {
      mean = StatUtils.mean(data)
      std = math.sqrt(StatUtils.variance(data, mean))
    }

    def transform(data: Array[Double]) = {
      data.map(e => (e - mean) / std)
    }

    def inverseTransform(data: Array[Double]) = {
      data.map(e => e * std + mean)
    }
  }

  class MaxAbsScaler extends Scaler {
    var absMax = 0.0
    def fit(data: Array[Double]): Unit = {
      absMax = data.max.abs
    }
    def transform(data: Array[Double]): Array[Double] = {
      data.map(_ / absMax)
    }

    def inverseTransform(data: Array[Double]): Array[Double] = {
      data.map(_ * absMax)
    }
  }
}
