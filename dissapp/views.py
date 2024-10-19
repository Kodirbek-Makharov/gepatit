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

def sirroz_haqida(request):
    return render(request, "sirroz-haqida.html")

def gepatit_davolash(request):
    return render(request, "gepatit-davolash.html")

def loyiha_haqida(request):
    return render(request, "loyiha-haqida.html")


def open_dataset(user, id=None):
    ds = Datasets.objects.filter(user=user, id=id).get()
    import pandas as pd
    import numpy as np
    bemorlar = Bemorlar.objects.filter(turi=1).all()
    # print(bemorlar)
    df = pd.DataFrame([o.__dict__ for o in bemorlar])
    # print (df.columns)
    df.drop(['_state', 'id', 'fio', 'created_at', 'turi', 'tekshiruv_fayli', 'doktor_id', 'izoh'], axis=1, inplace=True)

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
    # print(ds.file)
    # print(df)
    return df, ds


def load_excel(request):
    import pandas as pd
    import numpy as np

    df = pd.read_excel('uploads/gepatit.xlsx')

    for index, row in df.iterrows():
        model = Bemorlar()
        # print(row)
        model.fio = row['Ф.И.О']
        model.yosh = row['yoshi']
        model.pol = row['Пол']
        model.vdnk = row['В/ДНК']
        model.qHBsAg = row['qHBsAg']
        model.antihcv = row['Anti HCV']
        model.antihdv = row['AntiHDV']
        model.rnk_vgv = row['РНК ВГВ']
        model.rnk_vgd = row['РНК ВГД']
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
        # print(row['yoshi'])
    
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
    # print(df.head())
    # print(columns_all)

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

    import numpy as np
    from ai.cr2 import Stability

    df, ds = open_dataset(1, id)
    k1_value = 1
    k2_value = 2
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
    stabilities_to_db = ";".join(map(str, stabilities))

    df, ds = open_dataset(1, id)
    k1_value = 1
    k2_value = 3
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
    stabilities13, intervals, X_Binar = Stability(X, y, types)

    df, ds = open_dataset(1, id)
    k1_value = 2
    k2_value = 3
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
    stabilities23, intervals, X_Binar = Stability(X, y, types)

    stability_results = zip(stabilities,stabilities13,stabilities23, columns)

    stb = StabilityModel()
    stb.dataset = ds
    stb.stabilities = stabilities_to_db
    stb.save()

    fr = Feature_ranking()
    fr.dataset = ds
    fs = [f for s, f in sorted(zip(stabilities, columns), reverse=True)]
    fr.rankings = ";".join(fs)
    fr.save()

    nomlar = dict()
    nomlar['yosh'] = 'Yosh'
    nomlar['pol'] = 'Jins'
    nomlar['vdnk'] = 'HBV DNK (miqdoriy)'
    nomlar['qHBsAg'] = 'qHBsAg (miqdoriy)'
    nomlar['rnk_vgv'] = 'HBV RNK (sifat)'
    nomlar['rnk_vgd'] = 'HDV RNK (miqdoriy)'
    nomlar['gemoglabin'] = 'Gemoglobin'
    nomlar['eritrotsit'] = 'Eritrotsit'
    nomlar['rang'] = 'Rang'
    nomlar['trombosit'] = 'Trombotsit'
    nomlar['leykosit'] = 'Leykotsit'
    nomlar['sya'] = 'Segment yadroli neytrofill'
    nomlar['monotsit'] = 'Monotsit'
    nomlar['limfotsit'] = 'Limfotsit'
    nomlar['echt'] = 'ECHT'
    nomlar['alt'] = 'ALT'
    nomlar['ast'] = 'AST'
    nomlar['bilurbin'] = 'Bilirubin'
    nomlar['kreatinin'] = 'Kreatinin'
    nomlar['mochevina'] = 'Mochevina'
    nomlar['uzi'] = 'Ultra tovush tekshiruvi'
    nomlar['belok'] = 'Umumiy oqsil'
    nomlar['fibroskan'] = 'Fibroskan'
    nomlar['albumin'] = 'Albunim'
    nomlar['pvt'] = 'Virusga qarshi terapiya'
    nomlar['shf'] = 'Ishqoriy fosfotaza'
    nomlar['antihcv'] = 'antihcv'
    nomlar['antihdv'] = 'antihdv'

    return render(request, "stability-results.html", context={"stb": stability_results, "nomlar": nomlar })


def classification(request, id=None):

    id = 5
    k1_value = 1
    k2_value = 2
    propusk_task = "mean"

    import numpy as np
    from ai.cr2 import Stability
    from ai.crnulls import ClassificationAccuracyWorldMethods

    df, ds = open_dataset(1, id)

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
                # print(df.info())
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

    df, ds = open_dataset(1, 5)

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
            # print(col_mean, inds)
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
        # print("===================AJAX===================")
        q = json.load(request)
        # print(q)
        id = q.get("id")
        pt = q.get("pt")
        column_values = q.get("columns")

        df, ds = open_dataset(1, 5)

        if np.issubdtype(df[ds.class_column].dtype, np.integer):
            k1_value = int(k1_value)
            k2_value = int(k2_value)
        elif np.issubdtype(df[ds.class_column].dtype, np.str):
            k1_value = str(k1_value)
            k2_value = str(k2_value)

        df = df.loc[df[ds.class_column].isin([k1_value, k2_value])]
        fi = Feature_ranking.objects.filter(dataset=ds).get()

# yosh,jinsi,vdnk,qHBsAg,rnk,gemoglabin,eritrotsit,rang_kursatkich,trombotsit,
# leykotsit,cya,monotsit,limfotsit,echt,alt,ast,bilurbin,kreatinin,machevina,
# uzi,belok,fibroskan,albumin,pvt,shf,klass
def Saidov(yosh=1, jinsi=1, vdnk=1, qHBsAg=1, anti_hcv=1,anti_hdv=1,rnk_vgb=1,rnk_vgd=1, gemoglabin=1, eritrotsit=1, rang_kursatkich=1, trombotsit=1, leykotsit=1, cya=1,
              monotsit=1, limfotsit=1, echt=1, alt=1, ast=1, bilurbin=1, kreatinin=1, machevina=1, uzi=1, belok=1, fibroskan=1, albumin=1, pvt=1,
              shf=1, klass=1):

    result = dict()

#
    if shf > 2:
        result["shf"] = "shf chegaradan baland"
    else:
        result["shf"] = "shf chegaradan past"
    
#
    if pvt > 2:
        result["pvt"] = "pvt chegaradan baland"
    else:
        result["pvt"] = "pvt chegaradan past"
#++++++ norma 38-51
    if albumin > 51:
        result["albumin"] = "albumin normadan baland"
    elif albumin < 38:
        result["albumin"] = "albumin normadan past"
    else:
        result["albumin"] = "albumin normada"

# ++++++ afp norma 6-20
    # if afp > 20:
    #     result["afp"] = "AFP normadan baland"
    # elif albumin < 38:
    #     result["afp"] = "AFP normadan past"
    # else:
    #     result["afp"] = "AFP normada"

#
    if fibroskan > 2:
        result["fibroskan"] = "fibroskan chegaradan baland"
    else:
        result["fibroskan"] = "fibroskan chegaradan past"

#++++ oqsil norma 65-80
    if belok > 80:
        result["belok"] = "belok normadan baland"
    elif belok < 65:
        result["belok"] = "belok normadan  past"
    else:
        result["belok"] = "belok normada"
#
    if uzi > 2:
        result["uzi"] = "uzi chegaradan baland"
    else:
        result["uzi"] = "uzi chegaradan past"

#++++++ norma 1.7 - 8.3
    if machevina > 8.3:
        result["machevina"] = "machevina normadan baland"
    elif machevina < 1.7:
        result["machevina"] = "machevina normadan past"
    else:
        result["machevina"] = "machevina normada"

#++++ kreatinini erkan norma 53-97, ayol norma 42-80
    if jinsi == 1:
        if kreatinin > 97:
            result["kreatinin"] = "kreatinin normadan baland"
        elif kreatinin < 53:
            result["kreatinin"] = "kreatinin normadan past"
        else:
            result["kreatinin"] = "kreatinin normada"
    else:
        if kreatinin > 80:
            result["kreatinin"] = "kreatinin normadan baland"
        elif kreatinin < 42:
            result["kreatinin"] = "kreatinin normadan past"
        else:
            result["kreatinin"] = "kreatinin normada"


#+++++  norma 1.1 - 18.8
    if bilurbin > 18.8:
        result["bilurbin"] = "bilurbin normadan baland"
    elif bilurbin < 1.1:
        result["bilurbin"] = "bilurbin normadan past"
    else:
        result["bilurbin"] = "bilurbin norma"

#++++  norma erkak 37 eyol 31
    if jinsi == 1:
        if ast == 42:
            result["ast"] = "AST normada"
        else:
            result["ast"] = "AST normada emas"
    else:
        if ast == 32:
            result["ast"] = "AST normada"
        else:
            result["ast"] = "AST normada emas"


# ++++ norma ekrak 42  ayol 32
    if jinsi == 1:
        if alt == 42:
            result["alt"] = "ALT normada"
        else:
            result["alt"] = "ALT normada emas"
    else:
        if alt == 32:
            result["alt"] = "ALT normada"
        else:
            result["alt"] = "ALT normada emas"

#++++ norma erkak 2-10, ayol 2-15
    if jinsi == 1:
        if echt > 10:
            result["echt"] = "echt normadan baland"
        elif echt < 2:
            result["echt"] = "echt normadan past"
        else:
            result["echt"] = "echt normada"
    else:
        if echt > 15:
            result["echt"] = "echt normadan baland"
        elif echt < 2:
            result["echt"] = "echt normadan past"
        else:
            result["echt"] = "echt normada"
    
# +++++++ norma  19-37
    if limfotsit > 37:
        result["limfotsit"] = "limfotsit normadan baland"
    elif limfotsit < 19:
        result["limfotsit"] = "limfotsit normadan past"
    else:
        result["limfotsit"] = "limfotsit normada"


# +++++++++ norma 3.0 -11.0
    if monotsit >11.0 :
        result["monotsit"] = "monotsit normada baland"
    elif monotsit<3.0:
        result["monotsit"] = "monotsit normadan past"
    else:
        result["monotsit"] = "monotsit normada"

# segmeta yadroli neyrofil 47-72 norma
    if cya > 72:
        result["cya"] = "Segment yadroli neyrofil normadan baland"
    elif cya < 47:
        result["cya"] = "Segment yadroli neyrofil normadan  past"
    else:
        result["cya"] = "Segment yadroli neyrofil normada"

#+++++++  norma 4.0-9.0
    if leykotsit > 9.0:
        result["leykotsit"] = "leykotsit normadan baland"
    elif leykotsit < 4.0:
        result["leykotsit"] = "leykotsit normadan past"
    else:
        result["leykotsit"] = "leykotsit normada"

#+++++  norma  180-320
    if trombotsit > 320:
        result["trombotsit"] = "trombotsit normadan baland"
    elif trombotsit <180:
        result["trombotsit"] = "trombotsit normadan past"
    else:
        result["trombotsit"] = "trombotsit normada"

#+++++  rang 0.85-1.05 norma
    if rang_kursatkich > 1.05:
        result["rang_kursatkich"] = "rang ko'rsatkichi normadan baland"
    elif rang_kursatkich < 0.85:
        result["rang_kursatkich"] = "rang ko'rsatkichi normadan  past"
    else:
        result["rang_kursatkich"] = "rang ko'rsatkichi normada"

#++++++  # erkak 4-5, ayol 3.9-4.7 norma
    if jinsi == 1:
        if gemoglabin<4.0:
            result["eritrotsit"] = "eritrotsit normadan past"
        elif gemoglabin>5.0:
            result["eritrotsit"] = "eritrotsit normadan baland"
        else:
            result["eritrotsit"] = "eritrotsit normada"
    else:
        if gemoglabin < 3.9:
            result["eritrotsit"] = "eritrotsit normadan past"
        elif gemoglabin > 4.7:
            result["eritrotsit"] = "eritrotsit normadan baland"
        else:
            result["eritrotsit"] = "eritrotsit normada"

# ++++++ # erkak 130-160, ayol 120-140 norma
    if jinsi == 1:
        if gemoglabin<130:
            result["gemoglabin"] = "gemoglabin normadan past"
        elif gemoglabin>160:
            result["gemoglabin"] = "gemoglabin normadan baland"
        else:
            result["gemoglabin"] = "gemoglabin normada"
    else:
        if gemoglabin < 120:
            result["gemoglabin"] = "gemoglabin normadan past"
        elif gemoglabin > 140:
            result["gemoglabin"] = "gemoglabin normadan baland"
        else:
            result["gemoglabin"] = "gemoglabin normada"
#
    if rnk_vgb > 2:
        result["rnk_vgb"] = "rnk chegaradan baland"
    else:
        result["rnk_vgb"] = "rnk chegaradan past"
#
    if rnk_vgd > 2:
        result["rnk_vgd"] = "rnk chegaradan baland"
    else:
        result["rnk_vgd"] = "rnk chegaradan past"
#
    if qHBsAg > 2:
        result["qHBsAg"] = "qHBsAg chegaradan baland"
    else:
        result["qHBsAg"] = "qHBsAg chegaradan past"
#
    if yosh >= 30:
        result["yosh"] = "yoshi 30 dan katta"
    else:
        result["yosh"] = "yoshi 30 dan kichik"
#++++  # 1= erkak, 2-ayol
    if jinsi ==1 :
        result["jinsi"] = "Erkak"
    else:
        result["jinsi"] = "Ayol"
#
    if vdnk >2 :
        result["vdnk"]="vdnk chegaradan baland"
    else:
        result["vdnk"] = "vdnk chegaradan past"


    return  result


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
        id = q.get("id")
        fio = q.get("fio")
        k1_value = q.get("k1")
        k2_value = q.get("k2")
        bid = q.get("bid")
        pt = q.get("pt")
        column_values = q.get("columns")
        # print("====================================================")
        # print(column_values)
        def aniqlash(k1_value, k2_value):
            df, ds = open_dataset(1, 5)

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

            allowability=1
            alomat_soni = len(columns)
            olchangani = alomat_soni - column_values.count(None)
            if alomat_soni == olchangani:
                informativnost = 1
            else:
                fi = fi.rankings.split(";")
                D = 0
                for c, v in zip(columns, column_values):
                    r = fi.index(c)
                    if v is not None:
                        # D = D + alomat_soni - r
                        D = D + r + 1
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
                df = df.fillna(df.mean()).copy()
                X = df.loc[:, columns]  # alomatlar
                y = df[ds.class_column].values
                y = np.where(y == k1_value, 1, 2)
                x = np.array(column_values)
                for i in range(len(x)):
                    if x[i] is None:
                        x[i] = means[i]
                res, prb = ClassificationAccuracyWorldMethodsNewObject(X, y, x)
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
            sinf = round(sum(res) / len(res))
            probas = np.mean(prb, axis=0)
            sinf = k1_value if sinf==1 else k2_value
            return informativnost, res, probas[0].tolist(), sinf, allowability
        print(bid)
        if bid=="0":
            bemor = Bemorlar()
            print("nol")
        else:
            bemor = Bemorlar.objects.filter(id=bid).first()
            print("asd")
        print("==========", bemor)
        bemor.fio = fio
        bemor.yosh = column_values[0]
        bemor.pol = column_values[1]
        bemor.vdnk = column_values[2]
        bemor.qHBsAg = column_values[3]
        bemor.antihcv = column_values[4]
        bemor.antihdv = column_values[5]
        bemor.rnk_vgv = column_values[6]
        bemor.rnk_vgd = column_values[7]
        bemor.gemoglabin = column_values[8]
        bemor.eritrotsit = column_values[9]
        bemor.rang = column_values[10]
        bemor.trombosit = column_values[11]
        bemor.leykosit = column_values[12]
        bemor.sya = column_values[13]
        bemor.monotsit = column_values[14]
        bemor.limfotsit = column_values[15]
        bemor.echt = column_values[16]
        bemor.alt = column_values[17]
        bemor.ast = column_values[18]
        bemor.bilurbin = column_values[19]
        bemor.kreatinin = column_values[20]
        bemor.mochevina = column_values[21]
        bemor.uzi = column_values[22]
        bemor.belok = column_values[23]
        bemor.fibroskan = column_values[24]
        bemor.albumin = column_values[25]
        bemor.pvt = column_values[26]
        bemor.shf = column_values[27]
        bemor.doktor = request.user

        bemor.save()

        results = []
        informativnost, res, probas, sinf, allowability= aniqlash(1,2)
        # results.append([informativnost, res, probas, sinf, allowability])
        results.append({"informativnost": informativnost, "res": res, "probas": probas, "sinf": sinf, "allowability": allowability})
        informativnost, res, probas, sinf, allowability = aniqlash(1,3)
        results.append({"informativnost": informativnost, "res": res, "probas": probas, "sinf": sinf, "allowability": allowability})
        # results.append([informativnost, res, probas, sinf, allowability])
        informativnost, res, probas, sinf, allowability = aniqlash(2,3)
        results.append({"informativnost": informativnost, "res": res, "probas": probas, "sinf": sinf, "allowability": allowability})
        # results.append([informativnost, res, probas, sinf, allowability])

        #saidov_results= Saidov(*column_values)
        saidov_results= None
        # print(saidov_results)
        norma = dict()
        norma['yosh'] = ''
        norma['pol'] = ''
        norma['vdnk'] = '2'
        norma['qHBsAg'] = '2'
        norma['rnk_vgv'] = '0-2'
        norma['rnk_vgd'] = '2'
        norma['gemoglabin'] = 'Erkak: 130-160; Ayol: 120-140'
        norma['eritrotsit'] = 'Erkak: 4-5; Ayol: 3.9-4.7'
        norma['rang'] = '0.85 - 1.05'
        norma['trombosit'] = '180 - 320'
        norma['leykosit'] = '4.0 - 9.0'
        norma['sya'] = '47-72'
        norma['monotsit'] = 'Erkak: 3.0 - 11.0'
        norma['limfotsit'] = '19-37'
        norma['echt'] = 'Erkak: 2-10; Ayol: 2-15'
        norma['alt'] = 'Erkak: 42; Ayol: 32'
        norma['ast'] = 'Erkak: 37; Ayol: 31'
        norma['bilurbin'] = '1.1 - 18.8'
        norma['kreatinin'] = 'Erkak: 53-97; Ayol: 42-80'
        norma['mochevina'] = '1.7-8.3'
        norma['uzi'] = '0-2'
        norma['belok'] = '65-80'
        norma['fibroskan'] = '0-2'
        norma['albumin'] = '38-51'
        norma['pvt'] = '0-2'
        norma['shf'] = '0-2'

        nomlar = dict()
        nomlar['yosh'] = 'Yosh'
        nomlar['pol'] = 'Jins'
        nomlar['vdnk'] = 'HBV DNK (miqdoriy)'
        nomlar['qHBsAg'] = 'qHBsAg (miqdoriy)'
        nomlar['rnk_vgv'] = 'HBV RNK (sifat)'
        nomlar['rnk_vgd'] = 'HDV RNK (miqdoriy)'
        nomlar['gemoglabin'] = 'Gemoglobin'
        nomlar['eritrotsit'] = 'Eritrotsit'
        nomlar['rang'] = 'Rang'
        nomlar['trombosit'] = 'Trombotsit'
        nomlar['leykosit'] = 'Leykotsit'
        nomlar['sya'] = 'Segment yadroli neytrofill'
        nomlar['monotsit'] = 'Monotsit'
        nomlar['limfotsit'] = 'Limfotsit'
        nomlar['echt'] = 'ECHT'
        nomlar['alt'] = 'ALT'
        nomlar['ast'] = 'AST'
        nomlar['bilurbin'] = 'Bilirubin'
        nomlar['kreatinin'] = 'Kreatinin'
        nomlar['mochevina'] = 'Mochevina'
        nomlar['uzi'] = 'Ultra tovush tekshiruvi'
        nomlar['belok'] = 'Umumiy oqsil'
        nomlar['fibroskan'] = 'Fibroskan'
        nomlar['albumin'] = 'Albunim'
        nomlar['pvt'] = 'Virusga qarshi terapiya'
        nomlar['shf'] = 'Ishqoriy fosfotaza'

        bb = bemor.__dict__.copy()
        print(bb)
        bb.pop('_state')
        bb.pop('tekshiruv_fayli')
        return JsonResponse({'bid':bemor.pk,"results12": results[0], "results13": results[1], "results23": results[2], 'saidov':saidov_results, 'normalar':norma, 'bemord':bb, 'nomlar':nomlar})
        return JsonResponse({"informativnost": informativnost, "res": res, "probas": probas[0].tolist(), "sinf": sinf, "allowability": allowability})

    df, ds = open_dataset(1, id)

    # faqat tanlangan sinf qatorlarini qoldirish
    columns = df.columns.tolist()
    columns.remove(ds.class_column)

    return render(request, "tasniflash-new-object.html", context={"columns": columns, "k1": k1_value, "k2": k2_value, "pt": propusk_task, "id": id})

def new_objects_result_file(request):
    # print(request.FILES)
    # print(request.POST)
    from django.core.files.base import ContentFile
    from django.core.files import File

    i = request.POST.get('id')
    f = request.FILES.get('fayl')

    if i==0:
       return JsonResponse({'result':'Done!'})
    bemor = Bemorlar.objects.filter(pk=i).first()
    bemor.tekshiruv_fayli = f
    bemor.save()
    # print(bemor)
    # print(bemor.__dict__)
    # print("Saved")
    return JsonResponse({'result':'Done!'})

def bemorlar(request):
    from django.db.models import Q
    fish = request.GET.get('fish', None)
    if fish is None:
        bemorlar = Bemorlar.objects.filter(~Q(turi=1)).order_by('fio').all()
    else:
        bemorlar = Bemorlar.objects.filter(~Q(turi=1), fio__icontains=fish).order_by('fio').all()
    return render(request, 'bemorlar_list.html', context={'bemorlar':bemorlar})

def bemor(request, id):
    bemor = Bemorlar.objects.filter(pk=id).get()
    return render(request, 'bemor_info.html', context={'bemor':bemor})

def bemor_pdf(request, id):
    from django.http import FileResponse
    bemor = Bemorlar.objects.filter(pk=id).get()
    pdf = bemor.tekshiruv_fayli.path
    # response = HttpResponse(pdf.content, content_type='application/pdf')
    return FileResponse(open(pdf, 'rb'))

def bemor_set_class(request, id):
    from django.urls import reverse

    bemor = Bemorlar.objects.filter(pk=id).get()
    if request.method=="POST":
        klass = request.POST.get('klass')
        izoh = request.POST.get('izoh')
        # print("----------------------------------------")
        # print(klass, izoh, bemor)
        bemor.klass = klass
        bemor.izoh = izoh
        bemor.doktor = request.user
        bemor.save()
    return redirect(reverse('bemor_info', kwargs={'id':id}))
    # return render(request, 'bemor_set_class', context={'bemor':bemor})

def stat(request):
    # bemorlar o'g'il qiz soni
    o1 = Bemorlar.objects.filter(pol=1, klass=1).count()
    o2 = Bemorlar.objects.filter(pol=1, klass=2).count()
    o3 = Bemorlar.objects.filter(pol=1, klass=3).count()
    q1 = Bemorlar.objects.filter(pol=2, klass=1).count()
    q2 = Bemorlar.objects.filter(pol=2, klass=2).count()
    q3 = Bemorlar.objects.filter(pol=2, klass=3).count()

    y11 = Bemorlar.objects.filter(yosh__gte=0, yosh__lte=18, klass=1).count()
    y12 = Bemorlar.objects.filter(yosh__gte=0, yosh__lte=18, klass=2).count()
    y13 = Bemorlar.objects.filter(yosh__gte=0, yosh__lte=18, klass=3).count()
    y21 = Bemorlar.objects.filter(yosh__gte=19, yosh__lte=35, klass=1).count()
    y22 = Bemorlar.objects.filter(yosh__gte=19, yosh__lte=35, klass=2).count()
    y23 = Bemorlar.objects.filter(yosh__gte=19, yosh__lte=35, klass=3).count()
    y31 = Bemorlar.objects.filter(yosh__gte=36, yosh__lte=50, klass=1).count()
    y32 = Bemorlar.objects.filter(yosh__gte=36, yosh__lte=50, klass=2).count()
    y33 = Bemorlar.objects.filter(yosh__gte=36, yosh__lte=50, klass=3).count()
    y41 = Bemorlar.objects.filter(yosh__gte=51, yosh__lte=65, klass=1).count()
    y42 = Bemorlar.objects.filter(yosh__gte=51, yosh__lte=65, klass=2).count()
    y43 = Bemorlar.objects.filter(yosh__gte=51, yosh__lte=65, klass=3).count()
    y51 = Bemorlar.objects.filter(yosh__gte=66, yosh__lte=150, klass=1).count()
    y52 = Bemorlar.objects.filter(yosh__gte=66, yosh__lte=150, klass=2).count()
    y53 = Bemorlar.objects.filter(yosh__gte=66, yosh__lte=150, klass=3).count()

    return render(request, 'statistika.html', context={'soni': [o1, o2, o3, q1, q2, q3],
                                                       'yosh': [y11, y21, y31, y41, y51, y12, y22, y32, y42, y52, y13, y23, y33, y43, y53]})

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
