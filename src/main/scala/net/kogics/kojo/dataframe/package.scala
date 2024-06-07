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

import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.typeOf
import org.knowm.xchart.internal.chartpart.Chart
import org.knowm.xchart.internal.series.Series
import org.knowm.xchart.style.Styler
import tech.tablesaw.aggregate.AggregateFunctions.{max, mean, median, min, quartile1, quartile3, standardDeviation, stdDev}
import tech.tablesaw.api.CategoricalColumn
import tech.tablesaw.api.DoubleColumn
import tech.tablesaw.api.IntColumn
import tech.tablesaw.api.NumericColumn
import tech.tablesaw.api.Row
import tech.tablesaw.api.StringColumn
import tech.tablesaw.api.Table
import tech.tablesaw.columns.Column
import tech.tablesaw.io.csv.CsvReadOptions

package object dataframe {
  def readCsv(filename: String, separator: Char = ',', header: Boolean = true): Table = {
    val optionsBuilder =
      CsvReadOptions.builder(filename)
        .separator(separator) // table is tab-delimited
        .header(header)
    Table.read().csv(optionsBuilder.build())
  }

  def writeCsv(table: Table, filename: String): Unit = {
    table.write().csv(filename)
  }

  implicit class RichCategoricalColumn[T](c: CategoricalColumn[T]) {
    def percentByCategory() = {
      val catcnt = c.countByCategory
      val cats = catcnt.stringColumn(0)
      val cnts = catcnt.intColumn(1).asDoubleArray()
      val total = cnts.sum
      val percents = DoubleColumn.create("Percent", java.util.Arrays.stream(cnts.map(_ * 100 / total)))
      Table.create(c.name, cats, percents)
    }
  }

  implicit class RichNumericColumn[T <: Number](n: NumericColumn[T]) {
    def asDoubleSeq = n.asDoubleArray()
    def asIntSeq = asDoubleSeq.map(math.round(_).toInt)
  }

  implicit class RichStringColumn(s: StringColumn) {
    def asStringSeq = s.asObjectArray()
  }

  implicit class DataFrame(df: Table) {
    def length: Int = df.rowCount
    def head(n: Int = 5): Table = df.first(n)
    def tail(n: Int = 5): Table = df.last(n)
    def rows(n: Seq[Int]): Table = df.rows(n: _*)
    def columns[T: TypeTag](xs: Seq[T]): Table = {
      import scala.jdk.CollectionConverters._
      typeOf[T] match {
        case t if t =:= typeOf[Int] =>
          Table.create(df.name, df.columns(xs.asInstanceOf[Seq[Int]]: _*).asScala.toSeq: _*)
        case t if t =:= typeOf[String] =>
          Table.create(df.name, df.columns(xs.asInstanceOf[Seq[String]]: _*).asScala.toSeq: _*)
        case _ =>
          throw new RuntimeException("Invalid column index")
      }
    }
    def rowCols[T: TypeTag](rs: Seq[Int], cs: Seq[T]): Table = {
      columns(cs).rows(rs)
    }
    def describe(): Unit = {
      for (idx <- 0 until df.columnCount) {
        val column = df.column(idx)
        println("===")
        column match {
          case nc: NumericColumn[_] =>
            println(s"Column: ${column.name}")
            println(s"mean: ${mean.summarize(nc)}")
            println(s"std: ${stdDev.summarize(nc)}")
            println(s"min: ${min.summarize(nc)}")
            println(s"25%: ${quartile1.summarize(nc)}")
            println(s"50%: ${median.summarize(nc)}")
            println(s"75%: ${quartile3.summarize(nc)}")
            println(s"max: ${max.summarize(nc)}")

          case sc: StringColumn =>
            println(df.categoricalColumn(idx).countByCategory)

          case _ =>
        }
      }
    }

    def makeBarChart[A <: Styler, B <: Series](): Chart[A, B] = {
      import net.kogics.kojo.plot._
      val cnt = df.columnCount
      require(cnt == 1, "Dataframe should have only one column")
      val cc = df.categoricalColumn(0)
      val catcnt = cc.countByCategory
      barChart(" ", cc.name, "Percent", catcnt.stringColumn(0).asObjectArray().toIndexedSeq,
        catcnt.intColumn(1).asObjectArray().toIndexedSeq.map(_.toInt)).asInstanceOf[Chart[A, B]]
    }

    def makePieChart[A <: Styler, B <: Series](): Chart[A, B] = {
      import net.kogics.kojo.plot._
      val cnt = df.columnCount
      require(cnt == 1, "Dataframe should have only one column")
      val cc = df.categoricalColumn(0)
      val catcnt = cc.countByCategory
      pieChart(cc.name, catcnt.stringColumn(0).asObjectArray().toIndexedSeq,
        catcnt.intColumn(1).asObjectArray().toIndexedSeq.map(_.toInt)).asInstanceOf[Chart[A, B]]
    }

    def makeHistogram[A <: Styler, B <: Series](bins: Int = 10): Chart[A, B] = {
      import net.kogics.kojo.plot._
      val cnt = df.columnCount
      require(cnt == 1, "Dataframe should have only one column")
      val nc = df.numberColumn(0)
      histogram(nc.name, nc.name, "Count", nc.asDoubleArray(), bins).asInstanceOf[Chart[A, B]]
    }

    def makeLineChart[A <: Styler, B <: Series](): Chart[A, B] = {
      import net.kogics.kojo.plot._
      val cnt = df.columnCount
      require(cnt == 1 || cnt == 2, "Dataframe should have one or two columns")
      if (cnt == 1) {
        val nc2 = df.numberColumn(0)
        val chart = lineChart(" ", " ", nc2.name, Array.tabulate(10)(e => (e + 1).toDouble), nc2.asDoubleArray)
        chart.getStyler.setXAxisDecimalPattern("0")
        chart.getStyler.setXAxisMin(1.0)
        chart.asInstanceOf[Chart[A, B]]
      }
      else {
        val nc1 = df.numberColumn(0)
        val nc2 = df.numberColumn(1)
        lineChart(" ", nc1.name, nc2.name, nc1.asDoubleArray, nc2.asDoubleArray).asInstanceOf[Chart[A, B]]
      }
    }

    def makeScatterChart[A <: Styler, B <: Series](): Chart[A, B] = {
      import net.kogics.kojo.plot._
      val cnt = df.columnCount
      require(cnt == 1 || cnt == 2, "Dataframe should have one or two columns")
      if (cnt == 1) {
        val nc2 = df.numberColumn(0)
        val chart = scatterChart(" ", " ", nc2.name, Array.tabulate(10)(e => (e + 1).toDouble), nc2.asDoubleArray)
        chart.getStyler.setXAxisDecimalPattern("0")
        chart.getStyler.setXAxisMin(1.0)
        chart.asInstanceOf[Chart[A, B]]
      }
      else {
        val nc1 = df.numberColumn(0)
        val nc2 = df.numberColumn(1)
        scatterChart(" ", nc1.name, nc2.name, nc1.asDoubleArray, nc2.asDoubleArray).asInstanceOf[Chart[A, B]]
      }
    }

  }

  trait ColumnAdder[A] {
    def addColumn(df: Table, name: String)(filler: Row => A): Column[A]
  }

  object ColumnAdderInstances {

    def initColumn[A](df: Table, col: Column[A])(filler: Row => A): Column[A] = {
      df.forEach(row => col.appendMissing())
      df.forEach(row => col.set(row.getRowNumber, filler(row)))
      df.addColumns(col)
      col
    }

    implicit val stringColumn: ColumnAdder[String] =
      new ColumnAdder[String] {
        def addColumn(df: Table, name: String)(filler: Row => String): Column[String] = {
          val col = StringColumn.create(name)
          initColumn(df, col)(filler)
        }
      }

    implicit val intColumn: ColumnAdder[Int] =
      new ColumnAdder[Int] {
        def addColumn(df: Table, name: String)(filler: Row => Int): Column[Int] = {
          val col = IntColumn.create(name)
          initColumn[Int](df, col.asInstanceOf[Column[Int]])(filler)
        }
      }
  }

  object ColumnAdderSyntax {
    implicit class ColumnAdderOps[A](df: Table) {
      def addColumn(name: String)(filler: Row => A)(implicit ca: ColumnAdder[A]): Column[A] = {
        ca.addColumn(df, name)(filler)
      }
    }
  }
}
