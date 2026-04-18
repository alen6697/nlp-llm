SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    n.text
FROM
    `physionet-data.mimiciii_clinical.admissions` a
JOIN
    `physionet-data.mimiciii_notes.noteevents` n
ON
    a.hadm_id = n.hadm_id
WHERE
    n.category = 'Discharge summary'
    AND n.text IS NOT NULL;