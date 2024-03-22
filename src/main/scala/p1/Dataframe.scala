package p1

import net.kogics.kojo.dataframe._
import net.kogics.kojo.dataframe.ColumnAdderInstances._
import net.kogics.kojo.dataframe.ColumnAdderSyntax._
import net.kogics.kojo.plot._
import org.knowm.xchart.SwingWrapper
import tech.tablesaw.aggregate.AggregateFunctions.sum
import tech.tablesaw.api.Table

object Dataframe {
  def main(args: Array[String]): Unit = {
    val df = Table.read().csv("/home/lalit/Downloads/xAPI-Edu-Data.csv")
    println(df.structure)
    df.columns(Seq("gender"))
    df.head()
    df.stringColumn("gender").isMissing().size

    val cats = df.categoricalColumn("Topic").countByCategory.stringColumn("Category").asObjectArray().toIndexedSeq
    val counts =
      df.categoricalColumn("Topic").countByCategory.intColumn("Count").asObjectArray().toIndexedSeq.map(_.toInt)
    val chart = barChart("Subject Counts", "Subject", "Count", cats, counts)
//    new SwingWrapper(chart).displayChart()

    df.addColumn("Failed") { row => if (row.getString("Class") == "L") 1 else 0 }
    df.select("Class", "Failed")

    // group-by; split-apply-combine
    df.summarize("Failed", sum).by("Topic")
    // cross tab
    df.xTabCounts("Topic", "Class")
    // frequencies for categorical var
    df.categoricalColumn("Class").countByCategory

    val marks = readCsv("/home/lalit/work/kojo-ai/data/student-marks.csv")
    val info = readCsv("/home/lalit/work/kojo-ai/data/student-info.csv")
    marks.joinOn("Name").inner(info)
    marks.columns(Seq("Math")).makeLineChart()
  }
}
