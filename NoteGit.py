cd Ten_thu_muc: Di chuyển đến thư mục đó
cd .. : Quay lại thự mục cha
cd Ten_thu_muc1/Ten_thu_muc2: Di chuyển nhiều thư mục
dir : In ra các tên thư mục hiện tại
mkdir Ten_thu_muc: Tạo thư mục mới có tên là Ten_thu_muc
touch Ten_file: Tạo ra file mới có tên là Ten_file (Nếu Ten_file có dấu cách thì ghi là "Ten_file")
echo "Hello world" > a.txt: Ghi đè vào a.txt chuỗi
echo "Hello world" >> a.txt: Ghi newline vào a.txt chuỗi
cat ten_file: Hiện thị nội dung trong file
diff ten_file_1 ten_file_2: In ra nội dung khác nhau giữa 2 file
# Xóa
rm ten_file: xóa file
rm -d ten_thu_muc: xóa thư mục (thư mục rỗng)
rm -r ten_thu_muc: xóa thư mục (thư mục ko rỗng)


#----------------------------------------------
# Để trong [] là ko bắt buộc
git --help: Trợ giúp
git status: Hiện trạng thái repo
git log: Hiện thỉ lịch sử
git init [repo_name] : Tạo ra repo trống
git clone [repo_name] [clone_name]: tạo 1 bản sao được liên kết với repo
git config -l: xem cấu hình hiện tại của repo
#---------------------------
git config -l [--scope] [option_name] [value]:
scope: --system -> tất cả người dùng
       --global : liên quan đến toàn repo trên máy tính
       -- local : liên quan đến 1 repo

#---------Cấu hình thông tin
git config --global user.name "Ho va ten" : Cấu hình họ tên
git config --global user.email "email" : Cấu hình email
#---------------------------------

git add [filename(s)]: Thêm file vào Index
git add .: Thêm all file
git commit -m "Nội dung": Tạo commit lên repo
git status: Sự khác biệt 3 cây (HEAD - INDEX - Working diercty)
git diff: So sánh với commit cuối cùng
git log: Hiện lịch sử
git log --oneline: Hiện lịch sử trong 1 dòng
#--------------------------------------------
git init --bare [central_name]: Tạo 1 central repo
git clone [repo_name] [clone_name]: sao chép và liên kết repo_name
git fetch: Lấy các thông tin về commit mới từ central
gut pull: Lấy dữ liệu từ central về local repo
git push: Đẩy các commit từ local về central

#-------------------------------------------------
Cách sửa Conflict là sửa thủ công trong file bị Conflict
#------------------------------------------------
git checkout [commit_id]: Chuyển Head sang commit mà có commit_id (Hiểu đơn giản là chúng ta coi commit có commit_id là commit cuối cùng, những commit sau đó bị xóa đi)
                                                        (những commit bị xóa đi vẫn checkout đc !!!)
#-----------------------------------------------------
git branch <branch_name> : Tạo ra nhánh mới
git checkout <branch_name> : chuyển sang nhánh khác
git branch -l: Nhánh hiện tại và danh sách nhánh
#-----------------------------------------------------
git merge <branch_name> : Giả sử đang ở nhánh A thì sẽ merge nhánh branch_name vào nhánh A (merge theo thời gian,cái nào commit trước thì xếp trước)
git rebase A: Giả sử đang ở nhánh B thì sẽ nối nhánh A vào đầu nhánh B (A1 -> A2 -> A3 -> B1 -> B2, cái này là chuỗi của nhánh B)
#-------------------------------------------------------
git branch -d <branch_name>: Xóa nhánh branch_name (chú ý là chỉ xóa local chứ chưa xóa trên central)
Để xóa hẳn trên central hay remote thì dùng lệnh git push origin -d <branch_name>
git branch -a : Hiển thị danh sách nhánh ở central
#-----------------------------------------------
git clone link: Clone repo của người khác về máy