import random
import jieba

def gen_train_a_an(con, out_path):
    out = open(out_path,'a+',encoding='utf-8')
    replace_dict = {}
    replace_dict[' A '] = ' An '
    replace_dict[' a '] = ' an '
    replace_dict[' An '] = ' A '
    replace_dict[' an '] = ' a '
    for k,v in replace_dict.items():
        if k in con:
            if random.randint(1, 10) < 6 :
                con = str(con).replace(k,v)
                con_cut =jieba.cut(con)
                for c in con_cut:
                    if c == ' ':
                        continue

                    if c == v.replace(' ',''):
                        out.write(c+' '+'B-error'+'\n')
                        continue

                    if(',' in c and len(c)>1):
                        out.write(c.replace(',','')+' O\n, O\n')
                    else:
                        out.write(c+' O\n')
                out.write('\n')
            else:
                for c in con.split(' '):
                    out.write(c + ' O\n')
                out.write('\n')

            break
    out.close()

def get_count_A_An(con,count_dict):
    con_cut = jieba.cut(con)
    count = False
    for c in con_cut:
        if c == ' ':
            continue
        elif c in ['a','an']:
            count = True
        elif count:
            count_dict[c] = count_dict.get(c,0)+1
            count = False
    return count_dict



