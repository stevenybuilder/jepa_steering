import copy
import unittest

from offline_study.tasks.droid.droid_fit_inputs import select, validate_directory
from offline_study.tasks.droid.droid_catalogue import PREFIX


class DroidFitInputTests(unittest.TestCase):
    def row(self,index):
        return {'name':PREFIX+f'lab/{"success" if index%2 else "failure"}/date/{index}/trajectory.h5',
            'generation':'1710756229283371','size':'100','md5Hash':'AAAAAAAAAAAAAAAAAAAAAA=='}

    def test_selection_is_fixed_without_success_filter_and_order_independent(self):
        rows=[self.row(i) for i in range(74970)]
        chosen=select(rows)
        self.assertEqual(len(chosen),128);self.assertEqual(chosen,select(rows[::-1]))
        self.assertEqual({row['name'][len(PREFIX):].split('/')[1] for row in chosen},{'success','failure'})
        with self.assertRaises(ValueError):select(rows[:-1])
        with self.assertRaises(ValueError):select(rows[:-1]+rows[:1])

    def test_directory_requires_unique_metadata_and_original_generation(self):
        row=self.row(1);parent=row['name'].removesuffix('trajectory.h5')
        meta={**row,'name':parent+'metadata_episode.json','size':'20'}
        _,items,selected=validate_directory([row,meta],row)
        self.assertEqual(selected,meta)
        for changed in ([row],[row,meta,{**meta,'name':parent+'other.json'}],
                [{**row,'generation':'999'},meta],[row,meta,{**meta,'name':parent+'../unsafe.json'}]):
            with self.assertRaises(ValueError):validate_directory(changed,row)


if __name__=='__main__':unittest.main()
