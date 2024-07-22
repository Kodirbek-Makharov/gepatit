from django.shortcuts import redirect, render
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from .models import Datasets, Stability as StabilityModel, Feature_ranking, Classification, Allowability_intervals, Bemorlar
import json
from ai.dd import AcceptableInterval

# Create your views here.

#@login_required(login_url="/login", redirect_field_name=None)
def index(request):
    return render(request, "index.html")

def gepatit_haqida(request):
    return render(request, "gepatit-haqida.html")

def gepatit_davolash(request):
    return render(request, "gepatit-davolash.html")

def loyiha_haqida(request):
    return render(request, "loyiha-haqida.html")


def open_dataset(user, id=None):
    ds = Datasets.objects.filter(user=user, id=id).get()
    import pandas as pd
    import numpy as np
    bemorlar = Bemorlar.objects.filter(turi=1).all()

    df = pd.DataFrame([o.__dict__ for o in bemorlar])
    df.drop(['_state', 'id', 'fio', 'created_at', 'turi'], axis=1, inplace=True)

    return df, ds

def open_dataset_original(user, id=None):
    ds = Datasets.objects.filter(user=user, id=id).get()
    import pandas as pd
    import numpy as np

    if ds.ftype == "csv":
        df = pd.read_csv(ds.file)
    if ds.ftype == "xls":
        df = pd.read_excel(ds.file)
    if ds.ftype == "txt":
        df = pd.read_csv(ds.file)
    print(ds.file)
    print(df)
    return df, ds


def load_excel(user, id=None):
    import pandas as pd
    import numpy as np

    df = pd.read_excel('uploads/gepatit.xlsx')

    for index, row in df.iterrows():
        model = Bemorlar()
        print(row)
        model.fio = row['Ф.И.О']
        model.yosh = row['yoshi']
        model.pol = row['Пол']
        model.vdnk = row['В/ДНК']
        model.qHBsAg = row['qHBsAg']
        model.rnk = row['РНК']
        model.gemoglabin = row['гемоглабин']
        model.eritrotsit = row['Эритроцит']
        model.rang = row['ранг курсаткич']
        model.trombosit = row['Тромбоцит']
        model.leykosit = row['Лейкоцит']
        model.sya = row['с/я']
        model.monotsit = row['моноцит']
        model.limfotsit = row['лимфоцит']
        model.echt = row['ЭЧТ']
        model.alt = row['АЛТ']
        model.ast = row['АСТ']
        model.bilurbin = row['Билирубин']
        model.kreatinin = row['креатинин']
        model.mochevina = row['Мачевина']
        model.uzi = row['УЗИ'] # nominal
        model.belok = row['белок']
        model.fibroskan = row['Фиброскан']
        model.albumin = row['альбумин']
        model.pvt = row['ПВТ']
        model.shf = row['ШФ']
        model.klass = row['КЛАСС']
        model.save()
        print(row['yoshi'])
    
    # print(df)
    return HttpResponse("123")

@login_required(login_url="/login", redirect_field_name=None)
def datasets(request, id=None):
    if id is None:
        ds = Datasets.objects.filter(user=request.user).all()
        return render(request, "datasets.html", context={"datasets": ds})
    import numpy as np

    df, ds = open_dataset(request.user, id)

    columns = df.columns.tolist()
    columns_all = df.columns.tolist()
    if ds.class_column is None:
        ds.class_column = columns[-1]
        ds.class_column_values = ";".join(map(str, sorted(df[ds.class_column].unique())))
        ds.save()
    if ds.types is None:
        types = [0] * (len(columns) - 1)
        ds.types = ";".join(map(str, types))
        ds.save()
    if ds.contains_nans is None:
        df = df.replace(r"^\s*$", np.nan, regex=True)
        count_missing = df.isnull().sum().sum()
        # count_missing2 = df.isna().sum().sum()
        if count_missing > 0:
            ds.contains_nans = True
        else:
            ds.contains_nans = False
        ds.save()

    shape = df.shape

    columns.remove(ds.class_column)
    types = ds.types.split(";")
    types = [int(x) for x in types]
    columns = sorted(zip(columns, range(len(columns)), types))
    df10 = df.head(10)
    df10.reset_index(drop=True, inplace=True)
    df10.index = np.arange(1, len(df10) + 1)
    return render(request, "dataset.html", context={"dataset": ds, "columns": columns, "columns_all": columns_all, "shape": list(shape), "df": df10})


@login_required(login_url="/login", redirect_field_name=None)
def changetype(request, id, type_n):
    ds = Datasets.objects.filter(user=request.user, id=id).get()
    types = ds.types.split(";")
    types = [int(x) for x in types]
    types[type_n] = abs(1 - types[type_n])
    ds.types = ";".join(map(str, types))
    ds.save()
    return redirect("datasets", id=id)


@login_required(login_url="/login", redirect_field_name=None)
def change_class_column(request, id):
    if request.method == "POST":
        df, ds = open_dataset(request.user, id)
        # ds = Datasets.objects.filter(user=request.user, id=id).get()
        class_column = request.POST.get("column_name")
        ds.class_column = class_column
        ds.class_column_values = ";".join(map(str, sorted(df[ds.class_column].unique())))
        ds.save()
    return redirect("datasets", id=id)


def select_dataset_ajax(request, id=None):
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if is_ajax:
        # print("===================AJAX===================")
        id = json.load(request).get("id")
        df, ds = open_dataset(request.user, id)
        class_column_values = ds.class_column_values.split(";")
        contains_nans = ds.contains_nans
        return JsonResponse({"class_column_values": class_column_values, "contains_nans": contains_nans})


def stability(request, id=None):
    id = 5
    k1_value = 1
    k2_value = 2

    import numpy as np
    from ai.cr2 import Stability

    df, ds = open_dataset(3, id)

    # df.drop(['pvt', 'fibroskan', 'uzi'], axis=1, inplace=True)
    # df.replace(999, np.nan)
    # faqat tanlangan sinf qatorlarini qoldirish
    df = df.loc[df[ds.class_column].isin([k1_value, k2_value])]
    columns = df.columns.tolist()
    columns.remove(ds.class_column)
    X = df.loc[:, columns].to_numpy()  # alomatlar
    y = df[ds.class_column].values
    y = np.where(y == k1_value, 1, 2)
    types = ds.types.split(";")
    types = [int(x) for x in types]

    _K1 = "K1"
    _K2 = "K2"
    stabilities, intervals, X_Binar = Stability(X, y, types)
    stability_results = zip(stabilities, columns)
    stabilities_to_db = ";".join(map(str, stabilities))

    stb = StabilityModel()
    stb.dataset = ds
    stb.stabilities = stabilities_to_db
    stb.save()

    fr = Feature_ranking()
    fr.dataset = ds
    fs = [f for s, f in sorted(zip(stabilities, columns), reverse=True)]
    fr.rankings = ";".join(fs)
    fr.save()

    return render(request, "stability-results.html", context={"stb": stability_results})


def classification(request, id=None):

    id = 5
    k1_value = 1
    k2_value = 2
    propusk_task = "mean"

    import numpy as np
    from ai.cr2 import Stability
    from ai.crnulls import ClassificationAccuracyWorldMethods

    df, ds = open_dataset(3, id)

    # faqat tanlangan sinf qatorlarini qoldirish
    df = df.loc[df[ds.class_column].isin([k1_value, k2_value])]
    columns = df.columns.tolist()
    columns.remove(ds.class_column)
    X = df.loc[:, columns].to_numpy()  # alomatlar
    y = df[ds.class_column].values
    y = np.where(y == k1_value, 1, 2)
    types = ds.types.split(";")
    types = [int(x) for x in types]

    # null bor bo'lsa nima qilish
    if ds.contains_nans:
        df = df.replace(r"^\s*$", np.nan, regex=True)

        if propusk_task == "delete":
            if Classification.objects.filter(dataset=ds, type=propusk_task).count() == 0:
                print(df.info())
                # df = df.dropna().copy()
                # df.dropna(axis=0, inplace=True)

                # print(df.info())
                # print(df.shape)
                X = df.loc[:, columns]  # alomatlar
                res_latent = ClassificationAccuracyWorldMethods(X, y, s=0.2, r=5)
                cls = Classification()
                cls.dataset = ds
                cls.type = "delete"
                cls.nbayes = res_latent[0]
                cls.knn = res_latent[1]
                cls.svm = res_latent[2]
                cls.random_forest = res_latent[3]
                cls.dtree = res_latent[4]
                cls.lda = res_latent[5]
                cls.save()
        elif propusk_task == "mean":
            if Classification.objects.filter(dataset=ds, type=propusk_task).count() == 0:
                df = df.fillna(df.mean()).copy()
                X = df.loc[:, columns]  # alomatlar
                res_latent = ClassificationAccuracyWorldMethods(X, y, s=0.2, r=5)
                cls = Classification()
                cls.dataset = ds
                cls.type = "mean"
                cls.nbayes = res_latent[0]
                cls.knn = res_latent[1]
                cls.svm = res_latent[2]
                cls.random_forest = res_latent[3]
                cls.dtree = res_latent[4]
                cls.lda = res_latent[5]
                cls.save()
        elif propusk_task == "our":
            if Classification.objects.filter(dataset=ds, type=propusk_task).count() == 0:
                from ai.cr2 import Stability
                from ai.cr1 import RSByBinarFeature
                from ai.crnulls import LatentByStability

                types = ds.types.split(";")
                types = [int(x) for x in types]
                stabilities, intervals, X_Binar = Stability(X, y, ds.types)
                RS_binar, myu = RSByBinarFeature(X_Binar, y, stabilities)
                for k in range(2, X.shape[1] // 2 + 1):
                    X = LatentByStability(X_Binar, myu, stabilities, k, stability_border=0.55)
                    res_latent = ClassificationAccuracyWorldMethods(X, y, s=0.2, r=5)
                    cls = Classification()
                    cls.dataset = ds
                    cls.type = "our"
                    cls.k = k
                    cls.nbayes = res_latent[0]
                    cls.knn = res_latent[1]
                    cls.svm = res_latent[2]
                    cls.random_forest = res_latent[3]
                    cls.dtree = res_latent[4]
                    cls.lda = res_latent[5]
                    cls.save()

    _K1 = "K1"
    _K2 = "K2"
    # print("===========================")
    result = Classification.objects.filter(dataset=ds).all()
    return render(request, "tasniflash-results.html", context={"tasniflash": result})


def allowability(request, id=None):
    if request.method == "GET":
        ds = Datasets.objects.filter(user=request.user).all()
        return render(request, "allowability-select-dataset.html", context={"datasets": ds})

    import numpy as np

    id = request.POST.get("dataset")

    df, ds = open_dataset(3, 5)

    columns = df.columns.tolist()
    columns.remove(ds.class_column)
    X = df.loc[:, columns].to_numpy()  # alomatlar
    types = ds.types.split(";")
    types = [int(x) for x in types]

    indices = [i for i in range(len(types)) if types[i] == 0]  # miqdoriy

    Xm = X[:, indices].copy()
    columns = np.array(columns)[indices].tolist()

    for i in range(X.shape[1] - 1):
        for j in range(i + 1, X.shape[1]):
            x = X[:, i]
            y = X[:, j]

            allowable = Allowability_intervals.objects.filter(feature1=columns[i]).filter(feature2=columns[j])
            if len(allowable) > 0:
                continue
            col_mean = np.nanmean(x)
            inds = np.where(np.isnan(x))
            x[inds] = col_mean

            col_mean = np.nanmean(y, axis=0)
            inds = np.where(np.isnan(y))
            print(col_mean, inds)
            y[inds] = col_mean

            R, value_x, value_y = AcceptableInterval(x, y)

            dd = Allowability_intervals()
            dd.dataset = ds
            dd.feature1 = columns[i]
            dd.feature2 = columns[j]
            dd.left = R.min()
            dd.right = R.max()
            dd.median1 = value_x
            dd.median2 = value_y

            dd.save()
    dds = Allowability_intervals.objects.filter(dataset=ds).all()
    return render(request, "allowability-results.html", context={"dds": dds})


def check_allowability(request, id=None):
    import numpy as np
    import ai.dd

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if is_ajax:
        print("===================AJAX===================")
        q = json.load(request)
        print(q)
        id = q.get("id")
        pt = q.get("pt")
        column_values = q.get("columns")

        df, ds = open_dataset(3, 5)

        if np.issubdtype(df[ds.class_column].dtype, np.integer):
            k1_value = int(k1_value)
            k2_value = int(k2_value)
        elif np.issubdtype(df[ds.class_column].dtype, np.str):
            k1_value = str(k1_value)
            k2_value = str(k2_value)

        df = df.loc[df[ds.class_column].isin([k1_value, k2_value])]
        fi = Feature_ranking.objects.filter(dataset=ds).get()


def new_object(request, id=None):

    id = 5
    k1_value = 1
    k2_value = 2
    propusk_task = "mean"


    import numpy as np
    from ai.cr2 import Stability
    from ai.crnulls import ClassificationAccuracyWorldMethods, ClassificationAccuracyWorldMethodsNewObject

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    allowability = []
    if is_ajax:
        print("===================AJAX===================")
        q = json.load(request)
        print(q)
        id = q.get("id")
        k1_value = q.get("k1")
        k2_value = q.get("k2")
        pt = q.get("pt")
        column_values = q.get("columns")

        df, ds = open_dataset(3, 5)

        if np.issubdtype(df[ds.class_column].dtype, np.integer):
            k1_value = int(k1_value)
            k2_value = int(k2_value)
        elif np.issubdtype(df[ds.class_column].dtype, np.str):
            k1_value = str(k1_value)
            k2_value = str(k2_value)

        df = df.loc[df[ds.class_column].isin([k1_value, k2_value])]
        fi = Feature_ranking.objects.filter(dataset=ds).last()

        columns = df.columns.tolist()
        columns.remove(ds.class_column)

        """# allowability
        allowable = Allowability_intervals.objects.filter(dataset=ds).all()
        print(allowable)
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                if column_values[i] is None or column_values[j] is None:
                    continue
                allwb = allowable.filter(feature1=columns[i]).filter(feature2=columns[j]).get()
                print(allwb.feature1, allwb.feature2)
                R = column_values[i] / allwb.median1 - column_values[j] / allwb.median2
                if R < allwb.left or R > allwb.right:
                    allowability.append([i, j])"""
        allowability=1
        # informativnost
        alomat_soni = len(columns)
        olchangani = alomat_soni - column_values.count(None)
        if alomat_soni == olchangani:
            informativnost = 1
        else:
            fi = fi.rankings.split(";")
            D = 0
            print("FI = ", fi)
            for c, v in zip(columns, column_values):
                r = fi.index(c)
                if v is not None:
                    # D = D + alomat_soni - r
                    D = D + r + 1
                # print(c, v, r)
            alfa = olchangani * (olchangani + 1) / 2
            beta = olchangani * (2 * alomat_soni - olchangani + 1) / 2
            informativnost = 1 - ((D - alfa) / (beta - alfa)) * (1 - olchangani / alomat_soni)

            df = df.replace(r"^\s*$", np.nan, regex=True)

        if pt == "delete":
            X = df.loc[:, columns]  # alomatlar
            res_latent = ClassificationAccuracyWorldMethods(X, y, s=0.2, r=5)
        elif pt == "mean":
            print("-------------")
            means = df.mean().to_numpy()
            # print(means)
            df = df.fillna(df.mean()).copy()
            X = df.loc[:, columns]  # alomatlar
            y = df[ds.class_column].values
            y = np.where(y == k1_value, 1, 2)
            x = np.array(column_values)
            for i in range(len(x)):
                if x[i] is None:
                    # print(means[i])
                    x[i] = means[i]
            # print(X)
            res, prb = ClassificationAccuracyWorldMethodsNewObject(X, y, x)
            # print(res)
            # print(prb)
        elif pt == "our":
            from ai.cr2 import Stability
            from ai.cr1 import RSByBinarFeature
            from ai.crnulls import LatentByStability

            types = ds.types.split(";")
            types = [int(x) for x in types]
            stabilities, intervals, X_Binar = Stability(X, y, ds.types)
            RS_binar, myu = RSByBinarFeature(X_Binar, y, stabilities)
            for k in range(2, X.shape[1] // 2 + 1):
                X = LatentByStability(X_Binar, myu, stabilities, k, stability_border=0.55)
                res_latent = ClassificationAccuracyWorldMethods(X, y, s=0.2, r=5)

        def flatten_comprehension(matrix):
            return [int(item) for row in matrix for item in row]
        res = flatten_comprehension(res)
        # print(res, informativnost)
        sinf = round(sum(res) / len(res))
        #probalar
        probas = np.mean(prb, axis=0)
        print(probas)
        # print(probas[0][sinf-1])

        #sinfni nomiga o'tkazish
        sinf = k1_value if sinf==1 else k2_value
        # print(sinf)
        # sinf = sinf[2:]
        return JsonResponse({"informativnost": informativnost, "res": res, "probas": probas[0].tolist(), "sinf": sinf, "allowability": allowability})

    df, ds = open_dataset(3, id)

    # faqat tanlangan sinf qatorlarini qoldirish
    columns = df.columns.tolist()
    columns.remove(ds.class_column)

    return render(request, "tasniflash-new-object.html", context={"columns": columns, "k1": k1_value, "k2": k2_value, "pt": propusk_task, "id": id})



def login(request):
    if request.user.is_authenticated:
        return redirect("index")

    if request.method == "POST":
        from django.contrib.auth import login, authenticate  # add this

        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            messages.info(request, f"You are now logged in as {username}.")
            return redirect("index")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")


def logout(request):
    from django.contrib.auth import logout

    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect("index")
