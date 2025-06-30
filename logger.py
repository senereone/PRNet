import os
import codecs
import numpy as np
import pandas as pd

class Log(object):
    def __init__(self):
        pass

    def del_log(self, log_path):
        if os.path.exists(log_path):
            os.remove(log_path)

    def is_head_none(self, log_path):
        head = None
        if not os.path.exists(log_path):
            return True
        with open(log_path, 'r') as lines:
            for line in lines:
                head = line
                break
        if head is None:
            return True
        else:
            return False

    def line(self, values, names=None, name_value_sep=' ', sep='\t', value_pos=5):
        """
        trans values to a str line
        e.g.
            [1, 2, 3] => "1\t2\t3\t"
            [1, 2, 3], ["a", "b", "c"] => "a: 1\tb: 2\tc: 3"
        :param values: a list
        :param names: value name
        :param name_value_sep: the separator between name and value
        :param sep: the separator between name-value pair
        :param value_pos: round position
        :return: str line
        """
        if not isinstance(values, list):
            values = [values]
        # float to np.float32
        np_float32_values = [np.float32(v) if isinstance(v, float) else v for v in values]
        if names is None:
            out_list = ["%8.5f" % round(v, value_pos) if isinstance(v, np.float32) else str(v) for v in np_float32_values]
            out = sep.join(out_list)
        else:
            if not isinstance(names, list):
                names = [names]
            f = lambda x: "%s:%s%s" % (x[0], name_value_sep, str(round(x[1], value_pos)) if isinstance(x[1], np.float32) else str(x[1]))
            out = sep.join([f(x) for x in zip(names, np_float32_values)])
        return out

    def table(self, values, columns=None, value_pos=5):
        """
        trans list values to a table
        :param values: a list like [col0, col1, ...]
        :param columns: columns name
        :param value_pos: round value position
        :return: table
        """
        if len(np.array(values).shape) == 1:
            round_values = values
            try:
                round_values = np.round(values, value_pos)
            except:
                pass
            round_values = np.array(round_values)
        else:
            round_values = []
            for v in values:
                try:
                    v = np.round(v, value_pos)
                except:
                    pass
                round_values.append(v)
            round_values = np.array(round_values).transpose([1, 0])
        out = pd.DataFrame(round_values, columns=columns)
        return out

    def save_line(self, line, log_path):
        """
        add a line to log
        :param line: str line
        :param log_path: log path
        :return:
        """
        f = codecs.open(log_path, 'a', 'utf-8')
        f.write("%s\n" % line)
        f.close()

    def save_table(self, table, log_path, header=None):
        """
        add a table to log
        :param table: table
        :param log_path: log path
        :return:
        """
        if header is not None:
            f = codecs.open(log_path, 'w', 'utf-8')
            line = self.line(header)
            self.save_line(line, log_path)
            f.close()
        for row in range(len(table)):
            line = self.line(list(table.loc[row, :]))
            self.save_line(line, log_path)

    def clear(self, log_path):
        f = codecs.open(log_path, 'w', 'utf-8')
        f.close()

    def check_dir(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir



