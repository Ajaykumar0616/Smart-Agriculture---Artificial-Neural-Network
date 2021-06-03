from tkinter import *
import sub

def predict_crop():
    v.set(sub.clicked(e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get()))

root = Tk()
root.geometry('550x600')
root.title("Agriculture")

label_0 = Label(root, text="Crop Prediction System",width=20,font=("bold", 20))
label_0.place(x=90,y=53)


label_1 = Label(root, text="pH",width=20,font=("bold", 10))
label_1.place(x=80,y=130)

entry_1 = Entry(root, textvariable=e1)
entry_1.place(x=240,y=130)

label_2 = Label(root, text="N",width=20,font=("bold", 10))
label_2.place(x=80,y=160)

entry_2 = Entry(root, textvariable=e2)
entry_2.place(x=240,y=160)

label_3 = Label(root, text="P",width=20,font=("bold", 10))
label_3.place(x=80,y=190)

entry_3 = Entry(root, textvariable=e3)
entry_3.place(x=240,y=190)

label_4 = Label(root, text="K",width=20,font=("bold", 10))
label_4.place(x=80,y=220)

entry_4 = Entry(root, textvariable=e4)
entry_4.place(x=240,y=220)

label_5 = Label(root, text="Depth",width=20,font=("bold", 10))
label_5.place(x=68,y=250)

entry_5 = Entry(root, textvariable=e5)
entry_5.place(x=240,y=250)

label_6 = Label(root, text="T",width=20,font=("bold", 10))
label_6.place(x=80,y=280)

entry_6 = Entry(root, textvariable=e6)
entry_6.place(x=240,y=280)

label_7 = Label(root, text="Rainfall",width=20,font=("bold", 10))
label_7.place(x=60,y=310)

entry_7 = Entry(root, textvariable=e7)
entry_7.place(x=240,y=310)

Button(root, text='Submit',width=20,bg='brown',fg='white', command=predict_crop).place(x=180,y=380)

label_8 = Label(root, text="Predicted Crop",width=20,font=("bold", 10))
label_8.place(x=60,y=310)

v = StringVar()
e = Entry(root, textvariable=v, width=30, font=("bold", 20))
e.place(x=50,y=420)
w = StringVar()
e = Entry(root, textvariable=w, width=30, font=("bold", 20))
e.place(x=50,y=480)

v.set("predicted crop")
s = v.get()

w.set("Fertilizer Suggestion")

def predict():
    v.set(sub.clicked(e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get()))

print(s)

root.mainloop()