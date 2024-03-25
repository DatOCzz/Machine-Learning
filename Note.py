np.array([1,2,3]).T -> T ở đây là chuyển vị ma trận, lúc ý vector hàng sẽ thành vector cột

# plt
plt.plot(x,y): Sẽ hiện đồ thị đường ghấp khúc của các điểm trong 2 mảng x và y
plt.plot(x,y,'o'): Sẽ hiện các điểm của 2 mảng x và y (mặc định sẽ là màu xanh)
plt.plot(x,y,'ro'): Giống như trên nhưng là màu đỏ ('b^': màu xanh dương tam giác, 'go': màu xanh lục điểm, 'rs': màu đỏ hình chữ nhật)
plt.plot(X0[:,0],X0[:,1],'b^',markersize = 4, alpha = 0.3): markersize là kích thước của điểm, alpha là độ đậm của điểm (0<= alpha <= 1)
plt.axis([xmin, xmax, ymin, ymax]): Giới hạn đồ thị 
plt.xlabel('Height (cm)') , plt.ylabel('Weight (kg)'): Đặt tên trục
plt.show(): Show ra màn hình
plt.figure(figsize = (a, b)): Tạo khung với độ rộng là a và chiều cao là b
plt.axvline(a): Vẽ đường x = a
plt.axhline(a): Vẽ đường y = a
plt.axis("off"): Xóa các trục
plt.grid(False): Xóa lưới
plt.subplot(a,b,index): Tạo ra một bảng lưới có a hàng và b cột để vẽ nhiều đồ thị, index là ô mà vẽ đồ thị (bắt đầu từ 1)
plt.xticks([]): Tắt chỉ số trên trục x
plt.yticks([]): Tắt chỉ số trên trục y
plt.scatter(train_data, train_labels, c="b", s=4, label="Training data"): Vẽ các tọa độ (x,y) có màu là c, kích cỡ tọa độ là s và label
plt.imshow(train_images[i],cmap = plt.cm.viridis): show ra hình ảnh, cmap là màu sắc, mặc định là viridis
#-------------------------------------------------
X.shape[0] : Số hàng của X
X.shape[1] : Số cột của X
np.ones((shape, value)): Khởi tạo ma trận có kích cỡ là shape và giá trị của các phần tử sẽ là value
VD: np.ones(3,1) -> [1,1,1]
np.ones([2,2],1) -> [[1,1]
                    [1,1]]
np.concatenate((a, b), axis=0): Nối tiếp ma trận b vào a theo hàng
a = [1,2], b = [5,6]
    [3,4]
c = np.concatenate((a, b), axis=0) -> c = [1,2]
                                          [3,4]
                                          [5,6]
np.concatenate((a, b), axis=1): Nối tiếp ma trận b vào a theo cột
------------------------------------------------------
A =  np.array(_A),  A[a,b]: Phần tử hàng a cột b
                    A[:,b]: Lấy tất cả phần tử ở cột b
                    A[a,:]: Lấy tất cả phần tử ở hàng a

A.dot(B) hoặc A@B: Ma trận A nhân với vector B (Hoặc ma trận A nhân với ma trận B)
I = np.eye(a) : I là ma trận đơn vị kích cỡ là a
A*B: Từng phần tử của ma trận A nhân với phần tử tương ứng của ma trận B
A_i = np.linalg.pinv(A): Ma trận nghịch đảo của A
A_t = np.transpose(A): Ma trận chuyển vị
np.size(A,0): Trả về số hàng
np.size(A,1): Trả về số cột
np.sum(A,0) và np.sum(A,1): 0 là tính tổng theo cột, 1 là tính tổng theo hàng
np.max(A,0) và np.max(A,1): 0 là tính max theo cột, 1 là tính max theo hàng (tương tự như min)
A[a:b,c:d]: Lấy từ hàng a -> b-1, cột c -> d-1, nếu là A[a:b,:] thì là lấy tất cả các cột, tương tự cho hàng
a[a:b,c]: Lấy từ hàng a->b-1 và cột c, lưu ý nếu c = -1 thì nó sẽ là cột cuối cùng
np.amax(arr): Trả về giá trị max trong arr
np.zeros(shape): Tạo 1 ma trận có kích thước là shape, tất cả các phần tử = 0
np.random.choice(arr, size=None, replace=True): Chọn ngẫu nhiên 1 đoạn có kích thước là size trong mảng arr, nếu replace = True thì có thể có phần tử lặp
nếu = False thì ko có phần tử lặp
D = cdist(A, B): D là ma trận là độ dài của từng điểm của A với từng điểm của B 
np.argmin(a, axis=None, out=None): Đưa ra chỉ số của phần tử nhỏ nhất của mảng a
                                   Nếu a là mảng 1 chiều thì axis có thể để là None
                                   Nếu a là mảng 2 chiều thì axis = 0 sẽ tính theo cột và sẽ trả lại 1 mảng chứa các chỉ số nhỏ nhất theo cột, axis = 1 thì sẽ tính theo hàng
numpy.mean(a, axis=None, dtype=None, keepdims=False): Tính giá trị trung bình mảng a, axis = 0 thì tính theo cột nếu mảng 2 chiều, axis = 1 tính theo hàng mảng 2 chiều
s.strip(): Xóa bỏ khoảng trắng ở đầu và cuối của xâu s
# List
a.append(5): Thêm phần tử 5 vào cuối list
a.insert(1,"Hello"): Thêm chuỗi "Hello" vào vị trí 1 trong list
a.extend(b): thêm list b vào cuối list a
a.remove(5): Xóa phần tử ( truyền vào giá trị cần xóa chứ ko phải chỉ số)
a.pop(1): Xóa theo chỉ số (nếu là a.pop() thì sẽ xóa phần tử cuối cùng)
a.clear(): Xóa hết list
a.sort(): Sắp xếp a tăng dần
a.sort(reverse = True) : Sắp xếp giảm dần
a.sort(key = cmp): Sắp xếp theo cmp
a.sort(key = str.lower): Sắp xếp không phân biệt chữ hoa, chữ thường
a.reverse(): Đảo chiều list
b = a.copy(): copy a vào b
# Set
a.add(5): Thêm phần tử
a.update(b): Thêm b vào a
a.remove(...) hoặc a.discard(...): Xóa phần tử
a.clear(): Xóa a
c = a.union(b): Hợp của a và b (lưu vào c)
c = a.intersection_update(b): Giao của a và b (lưu vào c)
c = a.symmetric_difference(b): Chỉ lấy phần tử khác nhau giữa a và b
# Dict
x = a.keys(): Lấy ra tất cả khóa
x = a.values(): Lấy ra tất cả giá trị
x = a.items(): Lấy ra tất cả cặp khóa-giá trị
a.pop(5): Xóa giá trị khóa
a.popitem(): Xóa phần tử cuối
a.clear(): Xóa dict
a = b.copy(): copy a thành b
# OOP
self._age: protected (Thêm 1 dấu _ ở đầu thuộc tính)
self.__age: privated (thêm 2 dấu _ ở đầu thuộc tính)
# Xác suất
P(A|B) = P(A giao B)/ P(B)
P(A giao B)  = P(A|B)*P(B)
P(A giao B giao C) = P(A)*P(B|A)*P(C|AB)
P(A) = P(A giao B1) + P(A giao B2) + ... P(A giao Bn) (B1 + B2 + ... + Bn = omega)
    = P(A|B1)*P(B1) + P(A|B2)*P(B2) + ... + P(A|Bn)*P(Bn) 
Định lý Bayes: P(B|A) = P(A|B)*P(B)/P(A)
# np.array
b = a.copy(): Mảng b copy thành mảng a, thay đổi mảng a ko làm thay đổi mảng b và ngược lại
b = a.view(): Mảng b copy thành mảng a, thay đổi mảng a làm thay đổi mảng b và ngược lại
X.shape: Trả về 1 tuple chứa các chiều của mảng X
X.reshape(2,4): Giả sử X là mảng 1 chiều thì hàm reshape sẽ biến X thành mảng 2 chiều có 2 hàng và 4 cột
X.reshape(-1): Biến mảng nhiều chiều thành mảng 1 chiều
for x in np.nditer(arr): print(x): In ra từng phần tử trong mảng (đỡ phải dùng nhiều vòng for cho mảng nhiều chiều)
np.hstack((arr1, arr2)): Xếp arr2 vào arr1 theo hàng
np.vstack((arr1, arr2)): Xếp arr2 vào arr1 theo cột
np.dstack((arr1, arr2)): Xếp arr2 vào arr1 theo chiều sâu
np.array_split(arr,3): Chia mảng arr thành 3 nhóm
np.hsplit(arr, 3): Chia theo số hàng
np.vsplit(), np.dsplit(): chia theo số cột, chiều sâu
np.where(arr == 4): Trả về chỉ số mà giá trị tại đó bằng 4
np.searchsorted(arr, 7): Đưa ra chỉ số mà khi chèn số 7 vào thì mảng vẫn tăng dần (ban đầu mảng đc sắp xếp tăng dần)
np.sort(arr): Sắp xếp
np.median(arr): Giá trị ở giữa của mảng
np.std(arr): Độ lệch chuẩn
np.var(arr): Phương sai (Phương sai = (độ lệch chuẩn )^2)
np.percentile(arr, 50): Đưa ra giá trị X thỏa mãn, xác suất các phần tử xuất hiện trong mảng arr <= X là 50% 
np.random.uniform(a,b,k): Tạo ra 1 mảng số thực có k phần tử thuộc [a,b]
np.random.normal(a,b,k): a là giá trị trung bình, b là độ lệnh chuẩn, k là số lượng. Hàm này sẽ tạo ngẫu nhiên các giá trị xung quanh giá trị trung bình và có độ lệnh chuẩn là b
np.random.rand(d1,d2,...,dn): Tạo ra ma trận kích cỡ d1xd2x...xdn, các phần tử có giá trị thuộc (0,1)
np.random.randint(a,b,size = (d1,d2..)): Tạo ra ma trận kích thước d1xd2.. có giá trị là số nguyên trong [a,b-1]
np.random.randn(d0, d1, d2,..., dn): di là chiều,  mẫu ngẫu nhiên từ phân phối chuẩn hoá
np.linspace(a,b,c): Tạo ra 1 mảng có c phần tử,mỗi phần tử cách đều tăng dần từ a -> b

a.flatten(): Biến a thành 1 vector
a.ndim: Trả về 1 số nguyên là số chiều của mảng
a.shape: Trả vê 1 tuple là các chiều của mảng
a.dtype: Trả về kiểu phần tử trong mảng
Có thể khai báo a = np.array([1,2,3],dtype = "int16")
a.itemsize: Trả về số byte của kiểu phần tử
a.size: Số phần tử trong mảng
a.nbytes: Tổng số byte trong mảng
np.full((2,3),90): Tạo ra ma trận 2x3 có tất cả phần tử là 90
np.full_like(a,5): thay tất cả giá trj của a bằng 5
np.matmul(a,b): ma trận a nhân với ma trận b
np.linalg.det(a): Tính định thức ma trận a
np.linalg.norm(A, ord='fro'): Chuẩn Frobenious
np.trace(A): vết của ma trận
np.cov(x,y): Ma trận hiệp phương sai của chuỗi x và y
np.var(x, ddof=1): Phương sau của x
np.std(x, ddof=1): Độ lệch chuẩn của x
np.corrcoef(x, y): Hệ số tương quan của x và y.
np.sign(x): Hàm sign
np.array_equal(a,b): a bằng b thì trả về True, ngược lại là False



# Pandas
df = pd.read_csv("....csv"): Đọc file csv
df = pd.read_csv("...",sep = ",",header = 0,index_col = None)
sep: Là viết tắt của seperator, ký hiệu ngăn cách các trường trong cùng một dòng, 
     thường và mặc định là dấu phảy.

header: Mặc định là indice của dòng được chọn làm column name. 
        Thường là dòng đầu tiên của file. Trường hợp file không có header thì để header = None. 
        Khi đó indices cho column name sẽ được mặc định là các số tự nhiên liên tiếp từ 0 cho đến indice column cuối cùng.

index_col: Là indice của column được sử dụng làm giá trị index cho dataframe. 
           cột index phải có giá trị khác nhau để phân biệt giữa các dòng và khi chúng ta để index_col = None thì giá trị index sẽ được đánh mặc định từ 0 cho đến dòng cuối cùng.
df = pd.read_excel("....xlsx"): Đọc file excel

df.head(k): In ra k hàng đầu tiên

df.tail(k): In ra k hàng cuối cùng

df.sample(k): In ra k hàng ngẫu nhiên.

df.info():  cho ta biết định dạng và số lượng quan sát not-null của mỗi trường trong dataframe.

df.dtypes: Đưa ra kiểu dữ liệu của các cột

df.columns: Trả về 1 list chứa các tên của cột

df["Name"][0:5]: Lấy hàng 0 -> 4 của cột "Name"

df[["Name","Type 1"]][0:5]: Lấy hàng 0 -> 4 của cột "Name" và cột "Type 1"

df.iloc[a:b,c:d]: Lấy ra các hàng từ hàng a -> b-1, các cột tử c -> d - 1

df.iloc[a,b]: Đưa ra giá trị tại vị trí hàng a cột b

df.loc[df["Type 1"] == "Grass"]: In ra bảng mà có cột "Type 1" có giá trị bằng "Grass"

df.describe(): Đưa ra 1 bảng thống kê giá trị max, min, mean ... 

df.sort_values("Name"): Sort theo cột Name tăng dần (Nếu cột là các chuỗi thì sort theo thứ tự từ điển)

df.sort_values("Name",ascending = False): Sort giảm dần

df.sort_values(['medv', 'tax'], ascending = False).head(): sort theo nhiều cột

df = df.drop(columns = ["Name"]): Xóa cột "Name"

y_train = a.pop("Name": ): Xóa cột "Name" ở bảng a và lưu cột ý vào b

df["Name"].hist(): Biểu đồ cột

df["Name"].unique(): Đưa ra 1 list các giá trị khác nhau trong cột "Name"

df.select_dtypes('float'): Lọc ra các hàng có kiểu dữ diệu là float (integer, float, object, boolean).


df2.filter(regex='^age', axis=1): Lọc các cột mà có từ đầu là "age", axis=1 là làm việc với cột và axis=0 là làm việc với dòng

df["name"].min, df["name"].max, df["name"].mean(), df['tax'].median(), df['tax'].sum(): ......

df["Labels"] = pd.cut(df["tax"], bins = [100,200,300,1000], labels = ["Low","Median","High"]):
               Tạo ra cột mới tên là "Labels" có giá trị là các labels đc phân chia bởi
               100 - 200: Low
               200 - 300: Median
               300 - 1000: High
               ( Các giá trị ở cột "Tax")


df[["tax"]].apply(lambda x: x+1): Áp dụng hàm lambda cho tất cả các giá trị trong cột tax (Có thể cho nhiều cột)



dict_tax = {
      'low':'thap',
      'normal':'tb',        
      'high':'cao'
    }   
df['tax_labels'].map(dict_tax).head(): Thay thế các giá trị của cột tax_labels.


df["tax"].plot(): Vẽ đồ thị cột tax (Có thể cho nhiều cột)


df.melt(['Ho', 'Ten']): Giữ lại 2 cột Ho và Ten, các cột còn lại được định dạng bằng 2 cột là Variable và value.

pd.get_dummies(df): Chuyển về bảng one hot encoding

df_summary = df_iris[['Species', 'Petal.Length']].groupby('Species').mean(): Coi Species là các hàng, mỗi cột là Petal.Length

df.isnull(): Ô nào bị missing thì là True, còn ko bị missing thì là False.

# Tensorflow
# Khởi tạo giá trị
a = tf.Variable("hello",tf.string)
b = tf.Variable(3,tf.int32)
c = tf.Variable(3.14,tf.float64)
a = tf.constant([1,2,3])
# Phép tính
c = a + b hoặc c = tf.add(a,b)
c = a - b hoặc c = tf.subtract(a,b)
c = a/b hoặc c = tf.divide(a,b)
c = a*b hoặc c = tf.multiply(a,b) // Nếu là ma trận thì là tích Hadamard (nhân từng phần tử)
c = a@b (Nhân ma trận bình thường)
c = tf.tensordot(a,b,axes = 1) # dot product
b = a**2
#----------
tf.rank(a): In ra rand hoặc chiều
a.shape: Giống như numpy
tf.ones((3,3)): Giống như numpy (Có thể truyền list vào cx đc)
tf.range(a,b): Giống numpy
b = tf.reshape(a,(3,3)): Giống numpy (Có thể là tf.reshape(a,(3,-1)), -1 nghĩa là chương trình tự tính toán số chiều)
x = tf.cast(x,dtype = tf.float64): Chuyển đổi kiểu dữ liệu
a = tf.gather(a,indexs): index là 1 list, ví dụ index = [0,2] thì hàm gather sẽ đưa ra 1 list chứa các phần tử tại vị trí index
a_t = tf.transpose(a): Ma trận chuyển vị




keras.layers.Dropout(rate): Tỷ lệ (thường nằm trong khoảng từ 0 đến 1) xác định tỷ lệ các neuron sẽ bị "tắt".
VD:  Dropout(0.2),  # 20% của các neuron sẽ bị tắt ngẫu nhiên trong quá trình huấn luyện



model.evaluate(..., verbose)
verbose = 0: Mô hình sẽ chạy mà không hiển thị bất kỳ thông tin nào trên màn hình.
verbose = 1: Mô hình sẽ hiển thị thanh tiến trình (progress bar) trong quá trình đánh giá.
verbose = 2: Mô hình sẽ hiển thị một lần duy nhất kết quả của quá trình đánh giá, không có thanh tiến trình nào được hiển thị 




#  Sequential (1 input - 1 output)
model = keras.Sequential(
    [
        keras.Input(shape = (28*28)),
        layers.Dense(512, activation = "relu"),
        layers.Dense(256, activation = "relu"),
        layers.Dense(10),
    ]
)
print(model.summary())
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ["accuracy"],
)
model.fit(x_train,y_train,batch_size = 32, epochs = 5,verbose = 2)
model.evaluate(x_test,y_test,batch_size = 32, verbose = 2)

# Có thể thay bằng
model = keras.Sequential()
model.add(keras.Input(shape = (28*28)))
model.add(layers.Dense(512, activation = "relu"))
....


# functional API (more flexible than Sequential )
inputs = keras.Input(shape = (784))
x = layers.Dense(512, activation = "relu",name = "first_layer")(inputs)
x = layers.Dense(256,activation = "relu", name = "second_layer")(x)
outputs = layers.Dense(10,activation = "softmax")(x)
model = keras.Model(inputs = inputs, outputs = outputs)
model.summary()








# Pytorch
scalar = torch.tensor(7): Khởi tại 1 scalar(1 số nguyên)
scalar.ndim: Số chiều của scalar
a = scalar.item(): Chuyển từ kiểu tensor sang số nguyên (a = 7)
a.dtype : Kiểu dữ liệu của phần tử
a.device: a được lưu trữ ở đâu ? (thường là cpu hoặc gpu)
vector = torch.tensor([7, 7]): Khởi tạo 1 vector
vector.shape: Giống numpy, đưa ra các thành phần số chiều
random_tensor = torch.rand(size = (2,2)): Tạo ra 1 tensor ngẫu nhiên có kích cỡ là size, giá trị thuộc (0,1)
zeros = torch.zeros(size=(3, 4)): Tạo ra tensor chứa toàn số 0 với kích cỡ là size
ones = torch.ones(size=(3, 4)): Tạo ra tensor chứa toàn số 1 với kích cỡ là size
a = torch.arange(st,en,step): Tạo ra vector chứa các số từ st -> en với bước nhảy là step
tmp = torch.ones_like(input = ones): Tạo ra 1 ma trận chứa toàn số 1 có kích thước giống với input
tmp = torch.zeros_like(input = ones): .... 0
# Các phép toán
tensor = torch.tensor([1, 2, 3])
tensor + 10 : [11,12,13]
tensor*10 : [10,20,30]

a = torch.tensor([1,2,3])
b = torch.tensor([5,6,7])
a*b : Nhân từng phần tử -> [5,12,21]

# Nhân matrix
torch.matmul(a, b) (torch.mm for short) hoặc a@b
a.T : Chuyển vị ma trận a

a.min(): Giá trị nhỏ nhất
a.max(): Giá trị lớn nhất
a.type(torch.float32).mean(): Giá trị trung bình
a.sum(): Tổng giá trị
a.argmin(): Đưa ra chỉ số của giá trị nhỏ nhất
a.argmax(): Đưa ra chỉ số của giá trị lớn nhất


a = a.type(torch.float16): Đổi kiểu dữ liệu(float16, float32)

torch.reshape(a, shape): Thay đổi shape (hoặc a.reshape)
a = torch.stack([a,a,a,a],dim = 1): Hàm stack sẽ tạo ra 1 ma trận có 4 hàng là a nếu dim = 0, có 4 cột là a nếu dim = 1
a = a.squeeze(): Làm phẳng a, giả sử ban đầu a.shape = (1,4) thì a.shape sau sẽ biến thành vector 4 phần tử
a = a.unsqueeze(): Thêm 1 chiều vào, a = (7) -> a = (1,7)
a = a.permute(1,0,2): a.shape = (3,4,5) -> a.shape = (4,3,5) (các số 0,1,2 tượng trưng cho các chiều)
tensor = torch.from_numpy(array): Chuyển từ numpy sang pytorch (array là numpy)
b = a.numpy(): Chuyển từ pytorch sang numpy (a là pytorch)
len(a): Kích thước của a


# Ví dụ về một mô hình
#-----------------------------------------------------
class LinearRegression(nn.Module):
  def __init__(self):
      super().__init__()
      self.weights = nn.Parameter(torch.randn(1,dtype = torch.float,requires_grad= True))
      self.bias = nn.Parameter(torch.randn(1,dtype = torch.float,requires_grad = True))
      def forward(self,x:torch.tensor) -> torch.tensor:
        return self.weights*x + self.bias

#Hoặc
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
#-------------------------------------------------------
model = LinearRegression()
list(model.parameters()): Đưa ra các giá trị của parameter
model.state_dict(): Trạng thái của model

# predictions
with torch.inference_mode():
  y_predict = model(X_test)

# Khởi tạo loss và optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)


# Train Model
torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Check model device
next(model_1.parameters()).device: Kiểm tra device hiện tại của model
model_1.to(device) : Chuyển devicetor