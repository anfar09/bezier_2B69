import numpy as np

def get_variance(file_path, axis='X'):
    """
    อ่านไฟล์ .xyz แล้วคืนค่าความแปรปรวน (Variance) ของแกนที่เลือกกลับมาเป็นตัวเลขล้วนๆ
    
    :param file_path: ชื่อหรือที่ตั้งไฟล์ เช่น "Carhole5.xyz"
    :param axis: แกนที่ต้องการหาค่า ใส่ได้แค่ 'X', 'Y', หรือ 'Z'
    :return: ค่า Variance เป็นทศนิยม (float) หรือเป็น None ถ้าหาไฟล์ไม่เจอ
    """
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"Error: หาไฟล์ไม่เจอ -> {file_path}")
        return None

    # ดึงชิ้นส่วนข้อมูล
    points = data[:, :3] 

    # แปลงชื่อแกนที่พิมพ์เข้ามาเป็นตำแหน่งคอลัมน์ (0, 1, 2)
    axis = str(axis).upper()
    if axis == 'X':
        col_index = 0
    elif axis == 'Y':
        col_index = 1
    elif axis == 'Z':
        col_index = 2
    else:
        print(f"Error: ไม่รู้จักแกน '{axis}' (เลือกได้แค่ X, Y, Z)")
        return None

    # ดึงข้อมูลมาเฉพาะแนวตั้งของแกนนั้น แล้วลุยคำนวณ Variance
    axis_data = points[:, col_index]
    return np.var(axis_data)


def get_all_variances(file_path):
    """
    เหมือนด้านบน แต่จะเหมาดึงค่า Variance ของทั้ง 3 แกนมาพร้อมกัน
    เผื่ออยากเอาไปเขียนโค้ดเปรียบเทียบทีเดียว
    :return: โครงสร้างข้อมูล Dictionary แบบ {'X': 0.123, 'Y': 0.456, 'Z': 0.789}
    """
    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"Error: หาไฟล์ไม่เจอ -> {file_path}")
        return None

    points = data[:, :3] 
    return {
        'X': np.var(points[:, 0]),
        'Y': np.var(points[:, 1]),
        'Z': np.var(points[:, 2])
    }

# ==========================================
# ตัวอย่างการทำงานแบบโต้ตอบ (Interactive)
# บรรทัดข้างล่างนี้จะทำงานก็ต่อเมื่อกดรันไฟล์นี้โดยตรง
# หากถูกนำไปใช้ใน Web App โค้ดส่วนนี้ล่างนี้จะหลับไปครับ
# ==========================================
if __name__ == "__main__":
    print("✨ --- โปรแกรมคำนวณค่า Variance แกน 3 มิติ --- ✨")
    
    # ให้พิมพ์ถามหาชื่อไฟล์เลย
    user_file = input("👉 1. โปรดใส่ชื่อไฟล์ (ตัวอย่าง: /home/ken/Documents/2B/bazier/Carhole5.xyz): ")
    
    # ให้พิมพ์ถามหาแกน (ตอนนี้รองรับการพิมพ์หลายแกนพร้อมกัน)
    user_axis = input("👉 2. โปรดใส่แกนที่ต้องการ (ขอดูหลายแกนพิมพ์ X, Y, Z ได้เลย): ")
    
    print("\n⏳ กำลังประมวลผล...")
    
    # 1. ทำความสะอาดข้อความที่พิมพ์มาเผื่อคนพิมพ์ "X ,Z"
    # ทำการหั่นเชือกทิ้งตรงเครื่องหมายลูกน้ำ (,) แปลงเป็น List -> ['X', 'Z']
    # และใช้ .strip() เพื่อตัดช่องว่างที่คนอาจจะเผลอเคาะสเปซบาร์ทิ้งไว้
    axes_list = [a.strip() for a in user_axis.split(',')]
    
    # 2. เอาแต่ละแกนที่ตัดแบ่งแล้วมาเข้าลูป (Loop) ทีละรอบ
    for ax in axes_list:
        if ax == "": # ป้องกันกรณีพิมพ์ "X," แล้วมีค่าว่างเกินมา
            continue
            
        # เรียกใช้ฟังก์ชันด้านบนของเรา
        result = get_variance(file_path=user_file, axis=ax)
        
        # ถ้าดึงข้อมูลสำเร็จ (ไม่ Error)
        if result is not None:
            print(f"✅ สำเร็จ! ค่าความแปรปรวนของแกน {ax.upper()} คือ: {result}")
        
    print("-" * 50)