import pandas as pd


def iris_data(file):
    iris = pd.DataFrame(
        data=pd.read_csv(file, engine="python", names=["c1", "c2", "c3", "c4", "label"])
    )
    iris = iris[iris["label"] != "Iris-setosa"]
    iris.loc[iris["label"] == "Iris-versicolor", "label"] = 1
    iris.loc[iris["label"] == "Iris-virginica", "label"] = -1
    return iris


def hbt_data(file):
    hbt = pd.DataFrame(
        data=pd.read_csv(
            file, delimiter=r"\s+", engine="python", names=["c1", "label", "c2"]
        )
    )
    hbt = hbt[hbt.columns[[0, 2, 1]]]
    hbt.loc[hbt["label"] == 2, "label"] = -1
    return hbt


if __name__ == "__main__":
    print(iris_data("iris.data").shape)
    print(hbt_data("HC_Body_Temperature.txt").shape)
